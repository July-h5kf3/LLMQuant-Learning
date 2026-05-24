import torch
import torch.nn as nn


class DWConvBlock(nn.Module):
    """Depthwise separable 1D convolution block."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RelevanceCompressor(nn.Module):
    """
    Predict a probability distribution over visual tokens.

    Input:
        x: [B, Nv], first-layer text-to-visual attention proxy.
        mask: optional [B, Nv], true for valid positions.
    Output:
        [B, Nv] softmax distribution.
    """

    def __init__(self, channels: int = 32, num_blocks: int = 5) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.Sequential(*(DWConvBlock(channels) for _ in range(num_blocks)))
        self.output_proj = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, Nv], got {tuple(x.shape)}.")
        logits = self.input_proj(x.unsqueeze(1))
        logits = self.blocks(logits)
        logits = self.output_proj(logits).squeeze(1)
        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError(f"mask shape {tuple(mask.shape)} must match x shape {tuple(x.shape)}.")
            logits = logits.masked_fill(~mask.to(dtype=torch.bool, device=logits.device), torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)
