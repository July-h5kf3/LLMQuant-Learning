from __future__ import annotations

import torch

from prune_quant_baseline.scripts.run_infer_pruned import _generate_vanilla


class _Processor:
    def batch_decode(self, generated_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert generated_ids.tolist() == [[99]]
        return ["ok"]


class _ModelRejectingUnsupportedGenerateKwargs:
    generation_config = None

    def generate(self, **kwargs):
        if "mm_token_type_ids" in kwargs:
            raise ValueError("The following `model_kwargs` are not used by the model: ['mm_token_type_ids']")
        assert "input_ids" in kwargs
        assert "attention_mask" in kwargs
        return torch.tensor([[1, 2, 99]])


def test_generate_vanilla_filters_qwen25_processor_only_generate_kwargs() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 2, dtype=torch.long),
    }

    assert _generate_vanilla(_ModelRejectingUnsupportedGenerateKwargs(), _Processor(), inputs, 1) == "ok"
