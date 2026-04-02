import argparse
import contextlib
import os
import sys
import time
from functools import partial
from types import SimpleNamespace

import torch
import tqdm
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava1p5 import conversation as conversation_lib
from train_utils import (
    assert_no_meta_params,
    build_progress,
    get_device,
    get_next_batch,
    get_torch_dtype,
    log_train_metrics,
    move_batch_to_device,
    reset_meters,
    update_train_meters,
)
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, Summary

REQUIRED_CONFIG_KEYS = (
    "local_rank",
    "version",
    "vis_save_path",
    "precision",
    "image_size",
    "model_max_length",
    "lora_r",
    "vision_tower",
    "load_in_8bit",
    "load_in_4bit",
    "dataset",
    "sample_rates",
    "sem_seg_data",
    "refer_seg_data",
    "vqa_data",
    "reason_seg_data",
    "val_dataset",
    "dataset_dir",
    "log_base_dir",
    "exp_name",
    "epochs",
    "steps_per_epoch",
    "batch_size",
    "grad_accumulation_steps",
    "val_batch_size",
    "workers",
    "lr",
    "ce_loss_weight",
    "dice_loss_weight",
    "bce_loss_weight",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "explanatory",
    "beta1",
    "beta2",
    "num_classes_per_sample",
    "exclude_val",
    "no_eval",
    "vision_pretrained",
    "out_dim",
    "resume",
    "print_freq",
    "start_epoch",
    "gradient_checkpointing",
    "train_mask_decoder",
    "use_mm_start_end",
    "auto_resume",
    "conv_type",
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

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping.")

    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing_keys:
        raise KeyError(f"Missing config keys in YAML: {missing_keys}")

    config["config"] = config_path
    if parsed.local_rank is not None:
        config["local_rank"] = parsed.local_rank

    if config["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be one of: fp32, bf16, fp16")
    if config["conv_type"] not in {"llava_v1", "llava_llama_2"}:
        raise ValueError("conv_type must be one of: llava_v1, llava_llama_2")

    return SimpleNamespace(**config)


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
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    if args.seg_token_idx == tokenizer.unk_token_id:
        raise ValueError("[SEG] token was not added to the tokenizer correctly.")

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
        args.version, torch_dtype=torch_dtype, **model_args
    )
    assert_no_meta_params(
        model,
        module_keywords=["visual_model", "text_hidden_fcs", "mm_projector", "lm_head"],
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tokenizer_padding_side = tokenizer.padding_side

    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=get_device(args.local_rank))

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    if args.lora_r > 0:

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
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(
            x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
        ):
            p.requires_grad = True

    return tokenizer, model


def build_datasets(args, tokenizer):
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


def get_autocast_context(args, enabled):
    if not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    if args.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if args.precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def validate(val_loader, model_engine, epoch, writer, args):
    from utils.utils import AverageMeter, dict_to_cuda, intersectionAndUnionGPU

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        with torch.no_grad(), get_autocast_context(args, enabled=True):
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


def build_warmup_decay_scheduler(optimizer, total_steps, warmup_steps):
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps), 0)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = float(current_step - warmup_steps) / float(decay_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_torch(args, model, tokenizer, train_dataset, val_dataset):
    _, _, collate_fn = load_data_modules()
    device = get_device(args.local_rank)
    model.to(device=device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=build_collate_fn(args, tokenizer, collate_fn),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=build_collate_fn(args, tokenizer, collate_fn),
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=0.0,
    )
    scheduler = build_warmup_decay_scheduler(
        optimizer,
        total_steps=args.epochs * args.steps_per_epoch,
        warmup_steps=100,
    )

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "global_step": 0,
    }


def try_auto_resume(args):
    if not args.auto_resume or args.resume:
        return

    resume = os.path.join(args.log_dir, "ckpt_model")
    if os.path.exists(resume):
        args.resume = resume


def load_resume(args, state):
    if not args.resume:
        return

    checkpoint_path = args.resume
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location=state["device"])
    state["model"].load_state_dict(checkpoint["model"], strict=False)
    if "optimizer" in checkpoint:
        state["optimizer"].load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        state["scheduler"].load_state_dict(checkpoint["scheduler"])
    state["global_step"] = checkpoint.get("global_step", 0)
    args.start_epoch = checkpoint.get("epoch", 0)


def save_best_checkpoint(args, state, epoch, best_score, cur_ciou):
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

    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        trainable_names = {
            name for name, param in state["model"].named_parameters() if param.requires_grad
        }
        model_state = {
            name: tensor.detach().cpu()
            for name, tensor in state["model"].state_dict().items()
            if name in trainable_names
        }
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": state["global_step"],
                "model": model_state,
                "optimizer": state["optimizer"].state_dict(),
                "scheduler": state["scheduler"].state_dict(),
            },
            checkpoint_path,
        )


def train_one_epoch(args, state, epoch, writer, train_iter):
    batch_time, data_time, metric_meters, progress = build_progress(
        epoch, args.steps_per_epoch
    )

    state["model"].train()
    device = state.get("device")

    end = time.time()
    for global_step in range(args.steps_per_epoch):
        state["optimizer"].zero_grad(set_to_none=True)

        for _ in range(args.grad_accumulation_steps):
            input_dict, train_iter = get_next_batch(state["train_loader"], train_iter)
            data_time.update(time.time() - end)

            if device is not None:
                input_dict = move_batch_to_device(input_dict, device)
            else:
                from utils.utils import dict_to_cuda

                input_dict = dict_to_cuda(input_dict)
            with get_autocast_context(args, enabled=True):
                output_dict = state["model"](**input_dict)
                loss = output_dict["loss"] / args.grad_accumulation_steps
            update_train_meters(metric_meters, output_dict, input_dict["images"].size(0))
            loss.backward()

        state["optimizer"].step()
        state["scheduler"].step()
        state["global_step"] += 1

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0 and args.local_rank == 0:
            progress.display(global_step + 1)
            log_train_metrics(writer, metric_meters, batch_time, data_time, global_step)
            reset_meters(batch_time, data_time, *metric_meters.values())

        if global_step != 0 and writer is not None and args.local_rank == 0:
            curr_lr = state["scheduler"].get_last_lr()
            writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def train(args, state, writer):
    train_iter = iter(state["train_loader"])
    best_score, cur_ciou = 0.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train_one_epoch(args, state, epoch, writer, train_iter)

        is_best = False
        if state["val_loader"] is not None:
            giou, ciou = validate(
                state["val_loader"],
                state["model"],
                epoch,
                writer,
                args,
            )
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_best_checkpoint(args, state, epoch, best_score, cur_ciou)


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

    train_dataset, val_dataset = build_datasets(args, tokenizer)
    if val_dataset is not None:
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        print(f"Training with {len(train_dataset)} examples.")

    state = setup_torch(args, model, tokenizer, train_dataset, val_dataset)
    try_auto_resume(args)
    load_resume(args, state)
    train(args, state, writer)


if __name__ == "__main__":
    main(sys.argv[1:])
