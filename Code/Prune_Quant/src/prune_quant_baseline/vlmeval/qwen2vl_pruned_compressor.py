from __future__ import annotations

from typing import Any

import torch

from prune_quant_baseline.pruners.learned_compressor import LearnedCompressorPruner
from prune_quant_baseline.scripts.run_infer_pruned import (
    _build_pruned_generation_inputs,
    _generate_from_pruned_inputs,
    _move_inputs_to_model_device,
    _score_attention_proxy,
)
from .qwen2vl_pruned_gae import Qwen2VLPrunedGAE


class Qwen2VLPrunedCompressor(Qwen2VLPrunedGAE):
    """VLMEvalKit Qwen2-VL wrapper with learned compressor visual-token pruning."""

    def __init__(
        self,
        *args: Any,
        compressor_checkpoint: str,
        compressor_device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        device = compressor_device or str(next(self.model.parameters()).device)
        self._pq_pruner = LearnedCompressorPruner(compressor_checkpoint, device=device)

    def generate_inner(self, message: list[dict[str, str]], dataset: str | None = None) -> str:
        del dataset
        _, inputs = self._build_inputs(message)
        inputs = _move_inputs_to_model_device(self.model, inputs)
        meta = self._pq_adapter.get_visual_token_meta(self.model, inputs)
        scores = _score_attention_proxy(self.model, self._pq_pruner, inputs, meta)
        pruned_inputs, _, _ = _build_pruned_generation_inputs(
            model=self.model,
            adapter=self._pq_adapter,
            inputs=inputs,
            scores=scores,
            retention_ratio=self.retention_ratio,
            min_keep=self.min_keep,
        )
        response = _generate_from_pruned_inputs(
            model=self.model,
            processor=self.processor,
            pruned_inputs=pruned_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        torch.cuda.empty_cache()
        return response
