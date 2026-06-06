import unittest
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmllm.methods.qig.quantize.auto_scale import auto_scale_block


class TinySelfAttention(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, **kwargs):
        del kwargs
        return self.o_proj(self.v_proj(x))


class TinyMLP(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = TinySelfAttention(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = TinyMLP(hidden_size)


class QIGAutoScaleTest(unittest.TestCase):
    def test_weight_only_qwen2vl_scale_search_allows_missing_masks(self):
        if hasattr(auto_scale_block, "_layer_idx"):
            delattr(auto_scale_block, "_layer_idx")

        hidden_size = 4
        layer = Qwen2VLDecoderLayer(hidden_size)
        features = torch.randn(2, 3, hidden_size)
        input_feat = {
            "self_attn.q_proj": features.clone(),
            "self_attn.o_proj": features.clone(),
            "mlp.gate_proj": features.clone(),
            "mlp.down_proj": features.clone(),
        }

        scales = auto_scale_block(
            layer,
            module_kwargs={},
            w_bit=4,
            q_config={"zero_point": True, "q_group_size": -1},
            input_feat=input_feat,
            ans_mask=None,
            vis_mask=None,
            reweight_ratio_dict={"attn": None, "mlp": None},
        )

        self.assertEqual(len(scales), 4)
        for _, _, scale in scales:
            self.assertEqual(tuple(scale.shape), (hidden_size,))


if __name__ == "__main__":
    unittest.main()
