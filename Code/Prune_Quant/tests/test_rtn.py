import torch

from prune_quant_baseline.quant.rtn import fake_quantize_weight_rtn, rtn_fake_quant_linear_context


def test_fake_quantize_weight_rtn_preserves_shape_and_dtype() -> None:
    weight = torch.tensor([[0.0, 0.5, -1.0], [2.0, -2.0, 1.0]], dtype=torch.float16)

    quantized = fake_quantize_weight_rtn(weight, bits=4)

    assert quantized.shape == weight.shape
    assert quantized.dtype == weight.dtype


def test_rtn_fake_quant_linear_context_restores_forward() -> None:
    linear = torch.nn.Linear(2, 1, bias=False)
    original_forward = linear.forward

    with rtn_fake_quant_linear_context(linear, bits=4):
        assert linear.forward != original_forward

    assert linear.forward == original_forward
