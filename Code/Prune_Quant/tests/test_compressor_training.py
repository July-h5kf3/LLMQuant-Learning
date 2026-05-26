import torch

from prune_quant_baseline.compressors.conv1d_compressor import RelevanceCompressor
from prune_quant_baseline.compressors.train_compressor import _target_distribution
from prune_quant_baseline.pruners.learned_compressor import LearnedCompressorPruner


def test_target_distribution_keeps_top_half_and_sum_normalizes() -> None:
    oracle = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    mask = torch.ones_like(oracle, dtype=torch.bool)

    target = _target_distribution(oracle, mask, retention_ratio=0.5)

    assert torch.allclose(target, torch.tensor([[0.0, 0.0, 0.3 / 0.7, 0.4 / 0.7]]))


def test_learned_compressor_loads_checkpoint_metadata(tmp_path) -> None:
    model = RelevanceCompressor(channels=4, num_blocks=1)
    ckpt = tmp_path / "compressor.pt"
    torch.save({"model_state_dict": model.state_dict(), "channels": 4, "num_blocks": 1}, ckpt)

    pruner = LearnedCompressorPruner(ckpt)

    assert isinstance(pruner.compressor, RelevanceCompressor)
