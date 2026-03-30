import argparse
import os
import sys
from functools import partial

import torch
import tqdm
import transformers

try:
    import deepspeed
except ModuleNotFoundError:
    deepspeed = None

try:
    from peft import LoraConfig, get_peft_model
except ModuleNotFoundError:
    LoraConfig = None
    get_peft_model = None

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    Summary,
    intersectionAndUnionGPU,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Test Inference")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--test_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def get_device(args):
    if torch.cuda.is_available():
        return torch.device("cuda", args.local_rank)
    return torch.device("cpu")


def get_torch_dtype(precision):
    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.float16
    return torch_dtype


def move_batch_to_device(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(device=device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            input_dict[k] = [ele.to(device=device, non_blocking=True) for ele in v]
    return input_dict


def cast_batch_precision(input_dict, precision):
    if precision == "fp16":
        input_dict["images"] = input_dict["images"].half()
        input_dict["images_clip"] = input_dict["images_clip"].half()
    elif precision == "bf16":
        input_dict["images"] = input_dict["images"].bfloat16()
        input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
    else:
        input_dict["images"] = input_dict["images"].float()
        input_dict["images_clip"] = input_dict["images_clip"].float()


def load_val_modules():
    from utils.dataset import ValDataset, collate_fn

    return ValDataset, collate_fn


def build_model(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = get_torch_dtype(args.precision)
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=get_device(args))

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    if args.lora_r > 0:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError(
                "LoRA requires `peft`. Please install it in Code/LISA/.venv or set --lora_r 0."
            )

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        x not in name
                        for x in [
                            "visual_model",
                            "vision_tower",
                            "mm_projector",
                            "text_hidden_fcs",
                        ]
                    )
                    and any(x in name for x in lora_target_modules)
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def build_test_loader(args, tokenizer):
    ValDataset, collate_fn = load_val_modules()
    test_dataset = ValDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        args.test_dataset,
        args.image_size,
    )

    if args.batch_size != 1:
        raise ValueError("Test batch size must be 1 for segmentation evaluation.")

    sampler = None
    if deepspeed is not None and torch.cuda.device_count() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False, drop_last=False
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )
    return test_dataset, test_loader


def load_checkpoint_for_eval(model_or_engine, resume_path, device):
    if not resume_path:
        return

    if os.path.isdir(resume_path) and deepspeed is not None and hasattr(
        model_or_engine, "load_checkpoint"
    ):
        model_or_engine.load_checkpoint(resume_path)
        return

    checkpoint_path = resume_path
    if os.path.isdir(resume_path):
        checkpoint_path = os.path.join(resume_path, "checkpoint.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model_or_engine.load_state_dict(state_dict, strict=False)


def evaluate(test_loader, model_engine, args, device=None):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(test_loader):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if device is None:
            from utils.utils import dict_to_cuda

            input_dict = dict_to_cuda(input_dict)
        else:
            input_dict = move_batch_to_device(input_dict, device)
        cast_batch_precision(input_dict, args.precision)

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    print(
        f"Test dataset: {args.test_dataset}, samples: {len(test_loader.dataset)}, giou: {giou:.4f}, ciou: {ciou:.4f}"
    )
    return giou, ciou


def main(args):
    args = parse_args(args)
    tokenizer, model = build_model(args)
    _, test_loader = build_test_loader(args, tokenizer)
    device = get_device(args)

    if deepspeed is not None and torch.cuda.is_available():
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": 1,
            "fp16": {"enabled": args.precision == "fp16"},
            "bf16": {"enabled": args.precision == "bf16"},
            "zero_optimization": {"stage": 0},
        }
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        load_checkpoint_for_eval(model_engine, args.resume, device)
        evaluate(test_loader, model_engine, args)
    else:
        if deepspeed is None:
            print("DeepSpeed is not installed. Falling back to plain PyTorch evaluation.")
        model.to(device)
        load_checkpoint_for_eval(model, args.resume, device)
        evaluate(test_loader, model, args, device=device)


if __name__ == "__main__":
    main(sys.argv[1:])
