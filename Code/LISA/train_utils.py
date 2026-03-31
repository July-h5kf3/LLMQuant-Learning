import torch

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
