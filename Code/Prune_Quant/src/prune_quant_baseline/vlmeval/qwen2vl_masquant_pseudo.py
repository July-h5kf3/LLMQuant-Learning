from __future__ import annotations

from typing import Any

from .qwen2vl_pruned_gae import Qwen2VLPrunedGAE


class Qwen2VLMASQuantPseudo(Qwen2VLPrunedGAE):
    """VLMEvalKit wrapper that loads MASQuant pseudo-quantized Qwen2-VL."""

    def __init__(
        self,
        model_path: str,
        masquant_root: str,
        masquant_resume: str,
        model_type: str = "qwen2vl",
        masquant_act_scales: str | None = None,
        masquant_cmc_low_rank_adapters: str | None = None,
        masquant_cmc_white_matrix: str | None = None,
        masquant_cmc_rank: float = 0.2,
        masquant_cmc_quant_cmc: int = 0,
        retention_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            quant_method="masquant",
            masquant_root=masquant_root,
            masquant_resume=masquant_resume,
            masquant_act_scales=masquant_act_scales,
            masquant_cmc_low_rank_adapters=masquant_cmc_low_rank_adapters,
            masquant_cmc_white_matrix=masquant_cmc_white_matrix,
            masquant_cmc_rank=masquant_cmc_rank,
            masquant_cmc_quant_cmc=masquant_cmc_quant_cmc,
            retention_ratio=retention_ratio,
            **kwargs,
        )
