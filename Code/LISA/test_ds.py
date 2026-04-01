import argparse
import os
import sys
from functools import partial
from types import SimpleNamespace

import torch
import tqdm
import transformers
import yaml
from peft import LoraConfig, get_peft_model

from model.LISA import LISAForCausalLM
from model.llava1p5 import conversation as conversation_lib
from train_utils import cast_batch_precision, get_device, get_torch_dtype
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    Summary,
    intersectionAndUnionGPU,
)

REQUIRED_CONFIG_KEYS = (
    "local_rank",
    "version",
    "precision",
    "image_size",
    "model_max_length",
    "lora_r",
    "vision_tower",
    "load_in_8bit",
    "load_in_4bit",
    "dataset_dir",
    "test_dataset",
    "workers",
    "batch_size",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "ce_loss_weight",
    "dice_loss_weight",
    "bce_loss_weight",
    "vision_pretrained",
    "out_dim",
    "resume",
    "train_mask_decoder",
    "use_mm_start_end",
    "conv_type",
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Test Inference")
    parser.add_argument(
        "--config",
        default="configs/test_ds.yaml",
        type=str,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--local_rank",
        default=None,
        type=int,
        help="Optional runtime override for distributed launchers.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Optional checkpoint path override.",
    )
    parser.add_argument(
        "--test_dataset",
        default=None,
        type=str,
        help="Optional dataset override, e.g. ReasonSeg|val or ReasonSeg|test.",
    )
    parsed = parser.parse_args(args)

    config_path = parsed.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping.")

    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing_keys:
        raise KeyError(f"Missing config keys in YAML: {missing_keys}")

    config["config"] = config_path
    if parsed.local_rank is not None:
        config["local_rank"] = parsed.local_rank
    if parsed.resume is not None:
        config["resume"] = parsed.resume
    if parsed.test_dataset is not None:
        config["test_dataset"] = parsed.test_dataset

    if config["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be one of: fp32, bf16, fp16")
    if config["conv_type"] not in {"llava_v1", "llava_llama_2"}:
        raise ValueError("conv_type must be one of: llava_v1, llava_llama_2")

    return SimpleNamespace(**config)


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
    num_new_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    if args.seg_token_idx == tokenizer.unk_token_id:
        raise ValueError("[SEG] token was not added to the tokenizer correctly.")

    if args.use_mm_start_end:
        num_new_tokens += tokenizer.add_tokens(
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
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tokenizer_padding_side = tokenizer.padding_side

    vision_tower = model.get_model().get_vision_tower()
    if hasattr(vision_tower, "is_loaded") and not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch_dtype, device=get_device(args.local_rank))

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    if args.lora_r > 0 and args.resume:

        def find_linear_layers(model_obj, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model_obj.named_modules():
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

    if num_new_tokens > 0:
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
    if torch.cuda.device_count() > 1:
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

    if os.path.isdir(resume_path) and hasattr(model_or_engine, "load_checkpoint"):
        model_or_engine.load_checkpoint(resume_path)
        return

    checkpoint_path = resume_path
    if os.path.isdir(resume_path):
        checkpoint_path = os.path.join(resume_path, "checkpoint.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model_or_engine.load_state_dict(state_dict, strict=False)


def evaluate(test_loader, model_engine, args):
    from utils.utils import dict_to_cuda

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(test_loader):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)

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
    device = get_device(args.local_rank)
    model = model.to(device=device, dtype=get_torch_dtype(args.precision))
    load_checkpoint_for_eval(model, args.resume, device)
    evaluate(test_loader, model, args)


if __name__ == "__main__":
    main(sys.argv[1:])
