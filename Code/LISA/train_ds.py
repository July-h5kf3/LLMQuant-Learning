import argparse
import os
import shutil
import sys
import time
from functools import partial
from types import SimpleNamespace

import deepspeed
import torch
import tqdm
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter


from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    ProgressMeter,
    Summary,
)

DEFAULT_CONFIG = {
    "local_rank": 0,
    "version": "liuhaotian/llava-llama-2-13b-chat-lightning-preview",
    "vis_save_path": "./vis_output",
    "precision": "bf16",
    "image_size": 1024,
    "model_max_length": 512,
    "lora_r": 8,
    "vision_tower": "openai/clip-vit-large-patch14",
    "load_in_8bit": False,
    "load_in_4bit": False,
    "dataset": "sem_seg||refer_seg||vqa||reason_seg",
    "sample_rates": "9,3,3,1",
    "sem_seg_data": "ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
    "refer_seg_data": "refclef||refcoco||refcoco+||refcocog",
    "vqa_data": "llava_instruct_150k",
    "reason_seg_data": "ReasonSeg|train",
    "val_dataset": "ReasonSeg|val",
    "dataset_dir": "./dataset",
    "log_base_dir": "./runs",
    "exp_name": "lisa",
    "epochs": 10,
    "steps_per_epoch": 500,
    "batch_size": 2,
    "grad_accumulation_steps": 10,
    "val_batch_size": 1,
    "workers": 4,
    "lr": 0.0003,
    "ce_loss_weight": 1.0,
    "dice_loss_weight": 0.5,
    "bce_loss_weight": 2.0,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,v_proj",
    "explanatory": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "num_classes_per_sample": 3,
    "exclude_val": False,
    "no_eval": False,
    "vision_pretrained": "PATH_TO_SAM_ViT-H",
    "out_dim": 256,
    "resume": "",
    "print_freq": 1,
    "start_epoch": 0,
    "gradient_checkpointing": True,
    "train_mask_decoder": True,
    "use_mm_start_end": True,
    "auto_resume": True,
    "conv_type": "llava_v1",
}

TRAIN_METRIC_KEYS = (
    ("loss", "Loss"),
    ("ce_loss", "CeLoss"),
    ("mask_loss", "MaskLoss"),
    ("mask_bce_loss", "MaskBCELoss"),
    ("mask_dice_loss", "MaskDICELoss"),
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Finetuning")
    parser.add_argument(
        "--config",
        default="configs/train_ds.yaml",
        type=str,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--local_rank",
        default=None,
        type=int,
        help="Optional runtime override for distributed launchers.",
    )
    parsed = parser.parse_args(args)
    config_path = parsed.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    defaults = dict(DEFAULT_CONFIG)
    defaults.update(config)
    defaults["config"] = config_path
    if parsed.local_rank is not None:
        defaults["local_rank"] = parsed.local_rank

    if defaults["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be one of: fp32, bf16, fp16")
    if defaults["conv_type"] not in {"llava_v1", "llava_llama_2"}:
        raise ValueError("conv_type must be one of: llava_v1, llava_llama_2")

    return SimpleNamespace(**defaults)


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


def load_data_modules():
    from utils.dataset import HybridDataset, ValDataset, collate_fn

    return HybridDataset, ValDataset, collate_fn


def build_collate_fn(args, tokenizer, collate_fn):
    return partial(
        collate_fn,
        tokenizer=tokenizer,
        conv_type=args.conv_type,
        use_mm_start_end=args.use_mm_start_end,
        local_rank=args.local_rank,
    )


def build_progress(epoch, steps_per_epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    metric_meters = {
        key: AverageMeter(label, ":.4f") for key, label in TRAIN_METRIC_KEYS
    }
    progress = ProgressMeter(
        steps_per_epoch,
        [batch_time] + [metric_meters[key] for key, _ in TRAIN_METRIC_KEYS],
        prefix=f"Epoch: [{epoch}]",
    )
    return batch_time, data_time, metric_meters, progress


def reset_meters(*meters):
    for meter in meters:
        meter.reset()


def update_train_meters(metric_meters, output_dict, batch_size):
    for key, _ in TRAIN_METRIC_KEYS:
        metric_meters[key].update(output_dict[key].item(), batch_size)


def log_train_metrics(writer, metric_meters, batch_time, data_time, global_step):
    if writer is None:
        return

    for key, _ in TRAIN_METRIC_KEYS:
        writer.add_scalar(f"train/{key}", metric_meters[key].avg, global_step)
    writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
    writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)


def get_next_batch(train_loader, train_iter):
    try:
        input_dict = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        input_dict = next(train_iter)
    return input_dict, train_iter


def build_tokenizer_and_model(args):
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

    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(
            x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
        ):
            p.requires_grad = True

    return tokenizer, model


def build_datasets_and_loaders(args, tokenizer):
    HybridDataset, ValDataset, _ = load_data_modules()
    world_size = max(torch.cuda.device_count(), 1)
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
    )

    if not args.no_eval:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset


def validate(val_loader, model_engine, epoch, writer, args, device=None):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
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
            from utils.utils import intersectionAndUnionGPU

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

    if getattr(args, "distributed", False) and torch.distributed.is_initialized():
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0 and writer is not None:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")

    return giou, ciou


def train_with_deepspeed(args, model, tokenizer, train_dataset, val_dataset, writer):
    _, _, collate_fn = load_data_modules()
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, _, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=build_collate_fn(args, tokenizer, collate_fn),
        config=ds_config,
    )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        model_engine.load_checkpoint(args.resume)
        latest_path = os.path.join(args.resume, "latest")
        if os.path.exists(latest_path):
            with open(latest_path, "r", encoding="utf-8") as f:
                ckpt_dir = f.readlines()[0].strip()
            args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch

    val_loader = None
    if val_dataset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=build_collate_fn(args, tokenizer, collate_fn),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train_epoch_deepspeed(
            train_loader, model_engine, epoch, scheduler, writer, train_iter, args
        )

        is_best = False
        if val_loader is not None:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                os.makedirs(args.log_dir, exist_ok=True)
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        f"meta_log_giou{best_score:.3f}_ciou{cur_ciou:.3f}.pth",
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train_epoch_deepspeed(train_loader, model, epoch, scheduler, writer, train_iter, args):
    batch_time, data_time, metric_meters, progress = build_progress(
        epoch, args.steps_per_epoch
    )

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            input_dict, train_iter = get_next_batch(train_loader, train_iter)

            from utils.utils import dict_to_cuda

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            cast_batch_precision(input_dict, args.precision)

            output_dict = model(**input_dict)
            loss = output_dict["loss"]
            update_train_meters(metric_meters, output_dict, input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0 and args.local_rank == 0:
            progress.display(global_step + 1)
            log_train_metrics(writer, metric_meters, batch_time, data_time, global_step)
            reset_meters(batch_time, data_time, *metric_meters.values())

        if global_step != 0 and writer is not None and args.local_rank == 0:
            curr_lr = scheduler.get_last_lr()
            writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer, model = build_tokenizer_and_model(args)
    args.distributed = torch.cuda.device_count() > 1

    train_dataset, val_dataset = build_datasets_and_loaders(
        args, tokenizer
    )
    if val_dataset is not None:
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        print(f"Training with {len(train_dataset)} examples.")

    train_with_deepspeed(args, model, tokenizer, train_dataset, val_dataset, writer)


if __name__ == "__main__":
    main(sys.argv[1:])
