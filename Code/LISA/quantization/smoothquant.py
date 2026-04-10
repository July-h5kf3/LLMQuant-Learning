import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantization.pseudo_quant import PseudoQuantLinear
from quantization.quantization_utils import (
    QUANTIZED_LINEAR_NAMES,
    build_lisa_model,
    build_lisa_tokenizer,
    build_calibration_data,
    load_method_quant_config,
    pad_hidden_states,
    pad_position_ids,
    patch_transformers_compat,
    split_multimodal_calibration_inputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SmoothQuant activation scales for the LISA LLM backbone."
    )
    parser.add_argument(
        "--config",
        default="configs/quant/smoothquant.yaml",
        type=str,
        help="Path to the SmoothQuant YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate activation scales even if they already exist.",
    )
    return parser.parse_args()


def load_smoothquant_config(config_path):
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=(
            "model_path",
            "smoothquant_scale_path",
            "dataset_dir",
            "vision_tower",
            "vision_pretrained",
        ),
        bool_defaults={
            "keep_lm_export": False,
            "quantize_bmm_input": False,
        },
    )


def _default_scale_path(model_path):
    model_path = Path(model_path)
    return model_path.parent / f"{model_path.name}_smoothquant_act_scales.pt"


def _default_export_dir(scale_path):
    scale_path = Path(scale_path)
    return scale_path.parent / f"{scale_path.stem}_lm_backbone"


def _load_act_scales(scale_path):
    payload = torch.load(scale_path, map_location="cpu")
    if isinstance(payload, dict) and "act_scales" in payload:
        act_scales = payload["act_scales"]
    else:
        act_scales = payload

    if not isinstance(act_scales, dict) or not act_scales:
        raise ValueError(f"Invalid SmoothQuant scale payload: {scale_path}")

    return {
        key: value.cpu() if torch.is_tensor(value) else torch.as_tensor(value).cpu()
        for key, value in act_scales.items()
    }


def _save_act_scales(scale_path, model_path, config, act_scales):
    scale_path = Path(scale_path)
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "act_scales": act_scales,
            "meta": {
                "source_model_path": str(model_path),
                "scale_path": str(scale_path),
                "config_path": config["_config_path"],
                "alpha": config.get("alpha", 0.5),
                "w_bit": config.get("w_bit", 4),
                "a_bit": config.get("a_bit", 8),
                "num_scale_tensors": len(act_scales),
            },
        },
        scale_path,
    )


def _iter_batched_multimodal_inputs(multimodal_inputs, batch_size):
    if batch_size <= 0:
        raise ValueError("calibration_batch_size must be positive for SmoothQuant.")

    records = split_multimodal_calibration_inputs(multimodal_inputs)
    if not records:
        raise ValueError("SmoothQuant calibration inputs are empty.")

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        target_len = max(record["inputs_embeds"].shape[0] for record in chunk)

        batch = {
            "inputs_embeds": torch.stack(
                [
                    pad_hidden_states(record["inputs_embeds"].unsqueeze(0), target_len).squeeze(0)
                    for record in chunk
                ],
                dim=0,
            )
        }

        # `multimodal_inputs["attention_mask"]` stores the expanded causal mask
        # captured at the first decoder block, which is suitable for layer-wise
        # quantizers like AWQ/GPTQ. SmoothQuant replays the full decoder forward,
        # so we rebuild the standard 2D padding mask and let Llama expand it.
        batch["attention_mask"] = torch.stack(
            [
                torch.cat(
                    (
                        torch.ones(record["inputs_embeds"].shape[0], dtype=torch.bool),
                        torch.zeros(
                            target_len - record["inputs_embeds"].shape[0],
                            dtype=torch.bool,
                        ),
                    ),
                    dim=0,
                )
                for record in chunk
            ],
            dim=0,
        )

        if any(record.get("position_ids") is None for record in chunk):
            batch["position_ids"] = None
        else:
            batch["position_ids"] = torch.stack(
                [
                    pad_position_ids(record["position_ids"].unsqueeze(0), target_len).squeeze(0)
                    for record in chunk
                ],
                dim=0,
            )

        yield batch


def _get_runtime_device(runtime_device):
    if runtime_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(runtime_device))


def _get_decoder_layers(model_or_backbone):
    layers = getattr(model_or_backbone, "layers", None)
    if layers is not None:
        return layers

    nested_model = getattr(model_or_backbone, "model", None)
    if nested_model is not None and getattr(nested_model, "layers", None) is not None:
        return nested_model.layers

    raise TypeError("SmoothQuant expects a decoder backbone exposing `layers`.")


def _get_named_module_root(model_or_backbone):
    if getattr(model_or_backbone, "layers", None) is not None:
        return model_or_backbone
    nested_model = getattr(model_or_backbone, "model", None)
    if nested_model is not None and getattr(nested_model, "layers", None) is not None:
        return nested_model
    raise TypeError("SmoothQuant expects a decoder backbone exposing `layers`.")


@torch.no_grad()
def collect_smoothquant_act_scales(
    lisa_model,
    multimodal_inputs,
    *,
    runtime_device=None,
    calibration_batch_size=1,
    runtime_dtype=None,
):
    from model.llava1p5.model.language_model.llava_llama import LlavaLlamaForCausalLM

    act_scales = {}
    device = _get_runtime_device(runtime_device)
    if runtime_dtype is None:
        runtime_dtype = torch.float16 if device.type == "cuda" else torch.float32

    lisa_model.eval()
    lisa_model.to(device=device, dtype=runtime_dtype)
    module_root = _get_named_module_root(lisa_model.get_model())

    def stat_input_hook(_module, inputs, _output, name):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_dim).abs()
        incoming_max = hidden_states.max(dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.maximum(act_scales[name], incoming_max)
        else:
            act_scales[name] = incoming_max

    hooks = []
    for name, module in module_root.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(lambda m, x, y, n=name: stat_input_hook(m, x, y, n)))

    try:
        with torch.inference_mode():
            for batch in _iter_batched_multimodal_inputs(
                multimodal_inputs,
                calibration_batch_size,
            ):
                model_inputs = {
                    "inputs_embeds": batch["inputs_embeds"].to(device),
                    "use_cache": False,
                }
                if batch["attention_mask"] is not None:
                    model_inputs["attention_mask"] = batch["attention_mask"].to(device)
                if batch["position_ids"] is not None:
                    model_inputs["position_ids"] = batch["position_ids"].to(device)
                LlavaLlamaForCausalLM.forward(
                    lisa_model,
                    **model_inputs,
                )
    finally:
        for hook in hooks:
            hook.remove()
        lisa_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return act_scales


def _compute_smoothing_scales(act_scales, linears, alpha):
    if not isinstance(linears, (list, tuple)):
        linears = [linears]
    if not linears:
        raise ValueError("SmoothQuant expected at least one linear layer.")

    device = linears[0].weight.device
    dtype = linears[0].weight.dtype

    act_scales = act_scales.to(torch.float32)
    weight_scales = torch.cat(
        [linear.weight.detach().abs().amax(dim=0, keepdim=True).float() for linear in linears],
        dim=0,
    )
    weight_scales = weight_scales.amax(dim=0).clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1.0 - alpha)).clamp(min=1e-5)
    return scales.to(device=device, dtype=dtype)


@torch.no_grad()
def _smooth_norm_and_linears(norm, linears, act_scales, alpha):
    if not isinstance(linears, (list, tuple)):
        linears = [linears]
    scales = _compute_smoothing_scales(act_scales, linears, alpha)

    norm.weight.div_(scales)
    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(scales)

    for linear in linears:
        linear.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_llama_backbone(backbone_model, act_scales, alpha=0.5):
    layers = _get_decoder_layers(backbone_model)

    for layer_idx, layer in enumerate(layers):
        layer_name = f"layers.{layer_idx}"
        attn_scale_key = f"{layer_name}.self_attn.q_proj"
        mlp_scale_key = f"{layer_name}.mlp.gate_proj"

        if attn_scale_key not in act_scales:
            raise KeyError(f"Missing SmoothQuant activation scale: {attn_scale_key}")
        if mlp_scale_key not in act_scales:
            raise KeyError(f"Missing SmoothQuant activation scale: {mlp_scale_key}")

        _smooth_norm_and_linears(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            act_scales[attn_scale_key],
            alpha,
        )
        _smooth_norm_and_linears(
            layer.post_attention_layernorm,
            [
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
            ],
            act_scales[mlp_scale_key],
            alpha,
        )


def _resolve_parent_module(root_module, module_name):
    parent = root_module
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@torch.no_grad()
def pseudo_quantize_backbone_weight_act(
    backbone_model,
    *,
    w_bit,
    a_bit,
    quantize_bmm_input=False,
):
    layers = _get_decoder_layers(backbone_model)

    bmm_linear_names = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    }

    for layer in layers:
        for module_name in QUANTIZED_LINEAR_NAMES:
            parent, leaf = _resolve_parent_module(layer, module_name)
            module = getattr(parent, leaf)
            if not isinstance(module, nn.Linear):
                continue
            setattr(
                parent,
                leaf,
                PseudoQuantLinear.from_float(
                    module,
                    weight_quant="per_channel",
                    act_quant="per_token",
                    w_bit=w_bit,
                    a_bit=a_bit,
                    quantize_output=quantize_bmm_input and module_name in bmm_linear_names,
                ),
            )


def ensure_smoothquant_scales(
    model_path,
    quant_kwargs,
    *,
    force=False,
    runtime_device=None,
    runtime_dtype=None,
    lisa_model=None,
):
    patch_transformers_compat()

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs for SmoothQuant must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("SmoothQuant quant_kwargs must include '_config_path'.")

    config = load_smoothquant_config(config_path)
    scale_path = Path(config.get("smoothquant_scale_path", _default_scale_path(model_path)))
    if scale_path.exists() and not force:
        return str(scale_path)

    scale_device = _get_runtime_device(runtime_device)
    _, _, multimodal_inputs = build_calibration_data(str(model_path), config)
    owns_model = lisa_model is None

    try:
        if lisa_model is None:
            tokenizer = build_lisa_tokenizer(
                model_path,
                config["model_max_length"],
            )
            lisa_model = build_lisa_model(model_path, config, tokenizer)
        act_scales = collect_smoothquant_act_scales(
            lisa_model,
            multimodal_inputs,
            runtime_device=scale_device,
            calibration_batch_size=config.get("calibration_batch_size", 1),
            runtime_dtype=runtime_dtype,
        )
        _save_act_scales(scale_path, model_path, config, act_scales)
    finally:
        if owns_model:
            del lisa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return str(scale_path)


def load_smoothquant_backbone_into_lisa(
    lisa_model,
    model_path,
    quant_kwargs,
    *,
    force=False,
    device=None,
    torch_dtype=None,
):
    patch_transformers_compat()

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs for SmoothQuant must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("SmoothQuant quant_kwargs must include '_config_path'.")

    if torch_dtype is None:
        runtime_device = _get_runtime_device(device)
        torch_dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32

    config = load_smoothquant_config(config_path)
    scale_path = ensure_smoothquant_scales(
        model_path,
        quant_kwargs,
        force=force,
        runtime_device=device,
        runtime_dtype=torch_dtype,
        lisa_model=lisa_model,
    )
    act_scales = _load_act_scales(scale_path)

    smooth_llama_backbone(
        lisa_model.get_model(),
        act_scales,
        alpha=config.get("alpha", 0.5),
    )
    pseudo_quantize_backbone_weight_act(
        lisa_model.get_model(),
        w_bit=config.get("w_bit", 4),
        a_bit=config.get("a_bit", 8),
        quantize_bmm_input=config.get("quantize_bmm_input", False),
    )
    lisa_model.quantization_method = "smoothquant"
    lisa_model.config.quantization_config = {
        "quant_method": "smoothquant",
        "w_bit": config.get("w_bit", 4),
        "a_bit": config.get("a_bit", 8),
        "alpha": config.get("alpha", 0.5),
    }
    return lisa_model


def main():
    args = parse_args()
    config = load_smoothquant_config(args.config)
    tokenizer = build_lisa_tokenizer(
        config["model_path"],
        config["model_max_length"],
    )
    lisa_model = build_lisa_model(config["model_path"], config, tokenizer)
    scale_path = ensure_smoothquant_scales(
        model_path=config["model_path"],
        quant_kwargs={"_config_path": config["_config_path"]},
        force=args.force,
        runtime_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        lisa_model=lisa_model,
    )
    del lisa_model
    print(scale_path)


if __name__ == "__main__":
    main()
