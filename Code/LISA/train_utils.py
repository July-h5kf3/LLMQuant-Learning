import os

import numpy as np
import torch
from pycocotools import mask as mask_utils

from utils.utils import AverageMeter, ProgressMeter

TRAIN_METRIC_KEYS = (
    ("loss", "Loss"),
    ("ce_loss", "CeLoss"),
    ("mask_loss", "MaskLoss"),
    ("mask_bce_loss", "MaskBCELoss"),
    ("mask_dice_loss", "MaskDICELoss"),
)


def get_device(local_rank):
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
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


def assert_no_meta_params(model, module_keywords=None):
    if module_keywords is None:
        module_keywords = []

    meta_names = []
    for name, param in model.named_parameters():
        if getattr(param, "is_meta", False):
            if not module_keywords or any(keyword in name for keyword in module_keywords):
                meta_names.append(name)

    if meta_names:
        preview = ", ".join(meta_names[:10])
        if len(meta_names) > 10:
            preview += ", ..."
        raise RuntimeError(
            "Detected meta parameters after model loading. "
            "This usually means some weights were not materialized correctly. "
            f"Examples: {preview}"
        )


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


def encode_binary_mask(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_area_and_bbox(rle):
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()
    return area, bbox


def mask_score_from_logits(mask_logits):
    positive = mask_logits > 0
    if positive.any():
        return torch.sigmoid(mask_logits[positive]).mean().item()
    return torch.sigmoid(mask_logits).max().item()


def is_reasonseg_inst_dataset(dataset_name):
    return dataset_name in {"ReasonSegInst", "ReasonSeg-Inst", "ReasonInstanceSeg"}
