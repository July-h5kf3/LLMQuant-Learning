from pathlib import Path
from typing import Any, Optional

import torch

from prune_quant_baseline.compressors.conv1d_compressor import RelevanceCompressor
from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.attention_proxy import AttentionProxyPruner
from prune_quant_baseline.pruners.base import VisualTokenPruner


class LearnedCompressorPruner(VisualTokenPruner):
    """Wrap a trained relevance compressor as a visual-token pruner."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Compressor checkpoint does not exist: {self.checkpoint_path}")
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        channels = checkpoint.get("channels", 32) if isinstance(checkpoint, dict) else 32
        num_blocks = checkpoint.get("num_blocks", 5) if isinstance(checkpoint, dict) else 5
        self.compressor = RelevanceCompressor(channels=channels, num_blocks=num_blocks).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.compressor.load_state_dict(state_dict)
        self.compressor.eval()
        self.proxy = AttentionProxyPruner()

    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs: Any,
    ) -> torch.Tensor:
        proxy_scores = self.proxy.score(attentions=attentions, hidden_states=hidden_states, meta=meta)
        x = proxy_scores.to(self.device).unsqueeze(0)
        with torch.no_grad():
            scores = self.compressor(x).squeeze(0)
        return scores.to(proxy_scores.device)
