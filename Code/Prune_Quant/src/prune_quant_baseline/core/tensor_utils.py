from typing import Iterable

import torch


def ensure_long_tensor(values: Iterable[int] | torch.Tensor, device: torch.device | None = None) -> torch.LongTensor:
    """Convert values to a 1D long tensor."""

    tensor = values if isinstance(values, torch.Tensor) else torch.tensor(list(values))
    tensor = tensor.to(dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    if tensor.dim() != 1:
        raise ValueError(f"Expected a 1D tensor, got shape {tuple(tensor.shape)}.")
    return tensor
