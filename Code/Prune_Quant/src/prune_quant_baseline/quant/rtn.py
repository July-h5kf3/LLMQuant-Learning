from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quantize_weight_rtn(weight: torch.Tensor, *, bits: int = 4, group_size: int = 0) -> torch.Tensor:
    """Round-to-nearest symmetric fake quantization for Linear weights."""

    if bits < 2:
        raise ValueError("RTN bits must be >= 2.")
    if weight.dim() != 2:
        raise ValueError(f"RTN fake quant expects a 2D Linear weight, got {tuple(weight.shape)}.")

    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    compute = weight.detach().float()

    if group_size and group_size > 0:
        _, in_features = compute.shape
        chunks = []
        for start in range(0, in_features, group_size):
            chunk = compute[:, start : start + group_size]
            scale = chunk.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(chunk.dtype).eps) / qmax
            chunks.append((chunk / scale).round().clamp(qmin, qmax) * scale)
        return torch.cat(chunks, dim=1).to(dtype=weight.dtype, device=weight.device)

    scale = compute.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(compute.dtype).eps) / qmax
    return ((compute / scale).round().clamp(qmin, qmax) * scale).to(dtype=weight.dtype, device=weight.device)


def _is_linear_like(module: Any) -> bool:
    if isinstance(module, nn.Linear):
        return True
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight) or weight.dim() != 2:
        return False
    class_name = module.__class__.__name__.lower()
    return "linear" in class_name or hasattr(module, "forward_mas_infer")


@contextmanager
def rtn_fake_quant_linear_context(
    model: Any,
    *,
    bits: int = 4,
    group_size: int = 0,
    skip_name_parts: tuple[str, ...] = ("visual", "vision_tower", "multi_modal_projector", "lm_head", "audio"),
) -> Iterator[None]:
    """Temporarily run Linear-like modules with RTN fake-quantized weights."""

    patched: list[tuple[Any, Any]] = []

    def quantized_linear_forward(self: Any, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        weight = fake_quantize_weight_rtn(self.weight, bits=bits, group_size=group_size)
        bias = getattr(self, "bias", None)
        if not torch.is_tensor(bias):
            bias = None
        return F.linear(input, weight.to(dtype=input.dtype), bias.to(dtype=input.dtype) if bias is not None else None)

    try:
        for name, module in model.named_modules():
            if any(part in name for part in skip_name_parts):
                continue
            if not _is_linear_like(module):
                continue
            patched.append((module, module.forward))
            module.forward = MethodType(quantized_linear_forward, module)
        yield
    finally:
        for module, forward in patched:
            module.forward = forward
