from types import SimpleNamespace

import torch

from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter


class _Embedding:
    def __call__(self, input_ids):
        values = input_ids.to(torch.float32).unsqueeze(-1)
        return torch.cat([values, values], dim=-1)


class _TensorFeatureModel:
    config = SimpleNamespace(image_token_id=99, video_token_id=100)

    def get_input_embeddings(self):
        return _Embedding()

    def get_image_features(self, pixel_values, image_grid_thw):
        del pixel_values, image_grid_thw
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


class _KeywordRopeModel:
    config = SimpleNamespace(image_token_id=99)

    def get_rope_index(self, *, input_ids, image_grid_thw, video_grid_thw, attention_mask):
        del image_grid_thw, video_grid_thw
        assert attention_mask is not None
        return torch.arange(input_ids.shape[-1]).view(1, -1), torch.tensor([[0]])


def test_build_inputs_embeds_supports_tensor_image_features_without_placeholder_helper() -> None:
    adapter = Qwen2VLHFAdapter()
    inputs = {
        "input_ids": torch.tensor([[10, 99, 20, 99]]),
        "pixel_values": torch.ones(2, 3),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
    }

    embeds = adapter.build_inputs_embeds(_TensorFeatureModel(), inputs)

    assert embeds.shape == (1, 4, 2)
    assert embeds[0, 1].tolist() == [1.0, 2.0]
    assert embeds[0, 3].tolist() == [3.0, 4.0]


def test_build_position_ids_prefers_keyword_rope_index_call() -> None:
    adapter = Qwen2VLHFAdapter()
    inputs = {
        "input_ids": torch.tensor([[10, 99, 20]]),
        "attention_mask": torch.ones(1, 3),
    }

    position_ids = adapter.build_position_ids(_KeywordRopeModel(), inputs)

    assert position_ids.tolist() == [[0, 1, 2]]
    assert inputs["rope_deltas"].tolist() == [[0]]
