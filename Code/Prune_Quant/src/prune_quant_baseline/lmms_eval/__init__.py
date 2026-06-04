from __future__ import annotations


AVAILABLE_MODELS = {
    "prune_quant_qwen2vl": "Qwen2VLPrunedGAE",
}


def lmms_eval_model_manifests():
    from lmms_eval.models.registry_v2 import ModelManifest

    return [
        ModelManifest(
            model_id="prune_quant_qwen2vl",
            simple_class_path="prune_quant_baseline.lmms_eval.qwen2vl_pruned_gae.Qwen2VLPrunedGAE",
            aliases=("qwen2vl_pruned_gae",),
        )
    ]

