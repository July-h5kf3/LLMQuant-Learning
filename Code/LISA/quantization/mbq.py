import argparse
import copy
import gc
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.llava1p5.model.language_model.llava_llama import LlavaLlamaForCausalLM
from quantization.pseudo_quant import PseudoQuantLinear, pseudo_quantize_tensor
from quantization.quantization_utils import (
    QUANTIZED_LINEAR_NAMES,
    build_calibration_data,
    build_lisa_model,
    build_lisa_tokenizer,
    load_method_quant_config,
    patch_transformers_compat,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MBQ scales for the LISA LLM backbone.")
    parser.add_argument(
        "--config",
        default="configs/quant/mbq.yaml",
        type=str,
        help="Path to the MBQ YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate MBQ scales even if they already exist.",
    )
    return parser.parse_args()


def load_mbq_config(config_path):
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=(
            "model_path",
            "mbq_scale_path",
            "dataset_dir",
            "vision_tower",
            "vision_pretrained",
        ),
        bool_defaults={
            "keep_lm_export": False,
            "wa_quant": False,
            "reweight": False,
            "distort": False,
        },
    )


def _default_scale_path(model_path, *, wa_quant, w_bit, a_bit):
    model_path = Path(model_path)
    suffix = f"_mbq_w{w_bit}a{a_bit}" if wa_quant else f"_mbq_w{w_bit}"
    return model_path.parent / f"{model_path.name}{suffix}.pt"


def _load_mbq_results(scale_path):
    payload = torch.load(scale_path, map_location="cpu")
    if isinstance(payload, dict) and "scale" in payload:
        return payload
    raise ValueError(f"Invalid MBQ scale payload: {scale_path}")


def _save_mbq_results(scale_path, model_path, config, mbq_results):
    scale_path = Path(scale_path)
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale": mbq_results["scale"],
        "meta": {
            "source_model_path": str(model_path),
            "scale_path": str(scale_path),
            "config_path": config["_config_path"],
            "w_bit": config.get("w_bit", 4),
            "a_bit": config.get("a_bit", 8),
            "wa_quant": bool(config.get("wa_quant", False)),
            "zero_point": bool(config.get("zero_point", True)),
            "q_group_size": config.get("q_group_size", 128),
            "loss_mode": config.get("loss_mode", "mae"),
            "reweight": bool(config.get("reweight", False)),
            "distort": bool(config.get("distort", False)),
            "num_scale_tensors": len(mbq_results["scale"]),
        },
    }
    torch.save(payload, scale_path)


def _get_runtime_device(runtime_device):
    if runtime_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(runtime_device))


def _get_decoder_layers(backbone_model):
    layers = getattr(backbone_model, "layers", None)
    if layers is not None:
        return layers

    nested_model = getattr(backbone_model, "model", None)
    if nested_model is not None and getattr(nested_model, "layers", None) is not None:
        return nested_model.layers

    raise TypeError("MBQ expects a decoder backbone exposing `layers`.")


def _get_submodule(root_module, module_name):
    module = root_module
    for part in module_name.split("."):
        if not part:
            continue
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _set_submodule(root_module, module_name, new_module):
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_module
    else:
        setattr(parent, parts[-1], new_module)


def _is_norm_module(module):
    module_name = module.__class__.__name__.lower()
    return isinstance(module, nn.LayerNorm) or module_name.endswith("rmsnorm")


def _detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _to_runtime_tensor(value, device, runtime_dtype):
    if isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point:
            return value.to(device=device, dtype=runtime_dtype)
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_to_runtime_tensor(item, device, runtime_dtype) for item in value)
    if isinstance(value, list):
        return [_to_runtime_tensor(item, device, runtime_dtype) for item in value]
    if isinstance(value, dict):
        return {
            key: _to_runtime_tensor(item, device, runtime_dtype)
            for key, item in value.items()
        }
    return value


def _move_kwargs_to_device(kwargs, device, runtime_dtype):
    moved = {}
    for key, value in kwargs.items():
        moved[key] = _to_runtime_tensor(value, device, runtime_dtype)
    return moved


def _reshape_scales_for_input(scales, x):
    shape = [1] * x.ndim
    shape[-1] = scales.numel()
    return scales.view(*shape)


def _get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


def _build_search_scales(x_max, ratio):
    scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
    return scales / (scales.max() * scales.min()).sqrt()


def _compute_reconstruction_loss(
    org_out,
    out,
    *,
    ans_mask=None,
    vis_mask=None,
    reweight_ratio=None,
    loss_mode="mae",
):
    if loss_mode == "mse":
        diff = (org_out - out).float().pow(2)
    elif loss_mode == "mae":
        diff = (org_out - out).float().abs()
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    if ans_mask is not None and vis_mask is not None:
        ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(diff).to(diff.device)
        vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(diff).to(diff.device)
        masked_diff_ans = diff * ans_mask_expand
        masked_diff_vis = diff * vis_mask_expand
        if reweight_ratio is not None:
            if loss_mode == "mse":
                loss = (
                    masked_diff_ans.sum() / ans_mask_expand.sum().clamp(min=1)
                    + reweight_ratio
                    * (masked_diff_vis.sum() / vis_mask_expand.sum().clamp(min=1))
                )
            else:
                loss = (
                    masked_diff_ans.sum() + reweight_ratio * masked_diff_vis.sum()
                ) / (ans_mask_expand.sum() + vis_mask_expand.sum()).clamp(min=1)
        else:
            loss = diff.mean()
    elif ans_mask is not None and vis_mask is None:
        ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(diff).to(diff.device)
        masked_diff = diff * ans_mask_expand
        loss = masked_diff.sum() / ans_mask_expand.sum().clamp(min=1)
    else:
        loss = diff.mean()

    return float(loss.item())


def _scale_norm_linears(norm, linears, scales):
    if not isinstance(linears, (list, tuple)):
        linears = [linears]

    scales = scales.to(norm.weight.device)
    norm.weight.div_(scales)
    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(scales)

    for linear in linears:
        linear.weight.mul_(scales.view(1, -1).to(linear.weight.device))


def _scale_fc_fc(fc1, fc2, scales):
    scales = scales.to(fc1.weight.device)
    fc1.weight[-scales.numel() :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    fc2.weight.mul_(scales.view(1, -1).to(fc2.weight.device))


def _capture_first_layer_inputs(lisa_model, model_inputs, device, runtime_dtype):
    layers = _get_decoder_layers(lisa_model.get_model())
    original_layer = layers[0]
    captured_inputs = []
    captured_kwargs = {}
    sentinel = "__MBQ_CAPTURE__"

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            captured_inputs.append(inp.detach().cpu())
            captured_kwargs.clear()
            for key, value in kwargs.items():
                captured_kwargs[key] = _detach_to_cpu(value)
            raise RuntimeError(sentinel)

    layers[0] = Catcher(original_layer.to(device=device, dtype=runtime_dtype))
    forward_inputs = {
        key: _to_runtime_tensor(value, device, runtime_dtype)
        for key, value in model_inputs.items()
    }

    try:
        LlavaLlamaForCausalLM.forward(lisa_model, **forward_inputs)
    except RuntimeError as exc:
        if sentinel not in str(exc):
            raise
    finally:
        layers[0] = original_layer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not captured_inputs:
        raise ValueError("Failed to capture MBQ layer-0 inputs.")

    captured_kwargs["use_cache"] = False
    return captured_inputs[0], captured_kwargs


def _get_named_linears(module):
    return {name: child for name, child in module.named_modules() if isinstance(child, nn.Linear)}


def _cache_layer_inputs(layer, inps, layer_kwargs):
    input_feat = defaultdict(list)
    handles = []

    def cache_input_hook(_module, inputs, _output, name, feat_dict):
        feat_dict[name].append(inputs[0].detach().cpu())

    for name, module in _get_named_linears(layer).items():
        handles.append(
            module.register_forward_hook(
                lambda m, x, y, n=name, feat_dict=input_feat: cache_input_hook(
                    m, x, y, n, feat_dict
                )
            )
        )

    outputs = layer(inps, **layer_kwargs)[0]

    for handle in handles:
        handle.remove()

    input_feat = {name: torch.cat(values, dim=0) for name, values in input_feat.items()}
    return outputs, input_feat


def _pseudo_quantize_weight_for_search(weight, *, w_bit, q_config):
    return pseudo_quantize_tensor(
        weight,
        n_bits=w_bit,
        zero_point=q_config.get("zero_point", True),
        q_group_size=q_config.get("q_group_size", 128),
    )


def _search_module_scale_weight_only(
    block,
    linears,
    x,
    *,
    kwargs,
    w_bit,
    q_config,
    ans_mask,
    vis_mask,
    reweight_ratio,
    loss_mode,
):
    x = x.to(next(block.parameters()).device)
    with torch.no_grad():
        if isinstance(block, nn.Linear):
            org_out = block(x)
        else:
            org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = _get_act_scale(x)
    best_error = float("inf")
    best_scales = None
    org_sd = {name: value.detach().cpu() for name, value in block.state_dict().items()}

    for ratio_idx in range(20):
        scales = _build_search_scales(x_max, ratio_idx / 20.0)
        for linear in linears:
            linear.weight.mul_(scales.view(1, -1).to(linear.weight.device))
            quantized_weight = _pseudo_quantize_weight_for_search(
                linear.weight.data,
                w_bit=w_bit,
                q_config=q_config,
            )
            linear.weight.data = quantized_weight / scales.view(1, -1)

        if isinstance(block, nn.Linear):
            out = block(x)
        else:
            out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = _compute_reconstruction_loss(
            org_out,
            out,
            ans_mask=ans_mask,
            vis_mask=vis_mask,
            reweight_ratio=reweight_ratio,
            loss_mode=loss_mode,
        )
        if loss < best_error:
            best_error = loss
            best_scales = scales.detach().cpu()

        block.load_state_dict(org_sd)

    if best_scales is None:
        raise ValueError("MBQ failed to find a valid weight-only scale.")
    return best_scales


def _build_candidate_wa_linear(linear, *, w_bit, a_bit):
    return PseudoQuantLinear.from_float(
        linear,
        weight_quant="per_channel",
        act_quant="per_token",
        w_bit=w_bit,
        a_bit=a_bit,
        act_quant_mode="always",
    )


def _search_module_scale_weight_act(
    block,
    linears,
    block_linear_names,
    x,
    *,
    kwargs,
    w_bit,
    a_bit,
    ans_mask,
    vis_mask,
    reweight_ratio,
    loss_mode,
):
    x = x.to(next(block.parameters()).device)
    with torch.no_grad():
        if isinstance(block, nn.Linear):
            org_out = block(x)
        else:
            org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = _get_act_scale(x)
    best_error = float("inf")
    best_scales = None
    org_sd = {name: value.detach().cpu() for name, value in block.state_dict().items()}

    for ratio_idx in range(20):
        scales = _build_search_scales(x_max, ratio_idx / 20.0)
        scaled_x = x / _reshape_scales_for_input(scales.to(x.device), x)

        if isinstance(block, nn.Linear):
            linear = linears[0]
            linear.weight.mul_(scales.view(1, -1).to(linear.weight.device))
            quant_block = _build_candidate_wa_linear(linear, w_bit=w_bit, a_bit=a_bit)
            out = quant_block(scaled_x)
            del quant_block
        else:
            for linear, linear_name in zip(linears, block_linear_names):
                linear.weight.mul_(scales.view(1, -1).to(linear.weight.device))
                quant_linear = _build_candidate_wa_linear(linear, w_bit=w_bit, a_bit=a_bit)
                _set_submodule(block, linear_name, quant_linear)

            out = block(scaled_x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            for linear, linear_name in zip(linears, block_linear_names):
                _set_submodule(block, linear_name, linear)

        if isinstance(out, tuple):
            out = out[0]

        loss = _compute_reconstruction_loss(
            org_out,
            out,
            ans_mask=ans_mask,
            vis_mask=vis_mask,
            reweight_ratio=reweight_ratio,
            loss_mode=loss_mode,
        )
        if loss < best_error:
            best_error = loss
            best_scales = scales.detach().cpu()

        block.load_state_dict(org_sd)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_scales is None:
        raise ValueError("MBQ failed to find a valid weight-activation scale.")
    return best_scales


def _search_llama_layer_scales(
    layer,
    layer_kwargs,
    input_feat,
    *,
    w_bit,
    a_bit,
    q_config,
    wa_quant,
    reweight_ratio_dict,
    ans_mask,
    vis_mask,
    loss_mode,
):
    search_plan = [
        {
            "prev_op_name": "input_layernorm",
            "layer_names": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
            "block": layer.self_attn,
            "block_linear_names": ["q_proj", "k_proj", "v_proj"],
            "input_feat_key": "self_attn.q_proj",
            "reweight_ratio": reweight_ratio_dict["attn"],
            "enabled": True,
            "use_kwargs": True,
        },
        {
            "prev_op_name": "self_attn.v_proj",
            "layer_names": ["self_attn.o_proj"],
            "block": layer.self_attn.o_proj,
            "block_linear_names": None,
            "input_feat_key": "self_attn.o_proj",
            "reweight_ratio": reweight_ratio_dict["attn"],
            "enabled": layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape,
            "use_kwargs": False,
        },
        {
            "prev_op_name": "post_attention_layernorm",
            "layer_names": ["mlp.gate_proj", "mlp.up_proj"],
            "block": layer.mlp,
            "block_linear_names": ["gate_proj", "up_proj"],
            "input_feat_key": "mlp.gate_proj",
            "reweight_ratio": reweight_ratio_dict["mlp"],
            "enabled": True,
            "use_kwargs": False,
        },
        {
            "prev_op_name": "mlp.up_proj",
            "layer_names": ["mlp.down_proj"],
            "block": layer.mlp.down_proj,
            "block_linear_names": None,
            "input_feat_key": "mlp.down_proj",
            "reweight_ratio": reweight_ratio_dict["mlp"],
            "enabled": True,
            "use_kwargs": False,
        },
    ]

    scales_list = []
    for group in search_plan:
        if not group["enabled"]:
            continue

        linears = [_get_submodule(layer, name) for name in group["layer_names"]]
        block = group["block"]
        group_input = input_feat[group["input_feat_key"]].to(next(block.parameters()).device)
        block_kwargs = layer_kwargs if group["use_kwargs"] else {}

        if wa_quant:
            scales = _search_module_scale_weight_act(
                block,
                linears,
                group["block_linear_names"],
                group_input,
                kwargs=block_kwargs,
                w_bit=w_bit,
                a_bit=a_bit,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                reweight_ratio=group["reweight_ratio"],
                loss_mode=loss_mode,
            )
        else:
            scales = _search_module_scale_weight_only(
                block,
                linears,
                group_input,
                kwargs=block_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                reweight_ratio=group["reweight_ratio"],
                loss_mode=loss_mode,
            )

        scales_list.append(
            (
                group["prev_op_name"],
                tuple(group["layer_names"]),
                scales,
            )
        )

    return scales_list


@torch.no_grad()
def _apply_scales_to_module(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = _get_submodule(module, prev_op_name)
        layers = [_get_submodule(module, name) for name in layer_names]
        device = layers[0].weight.device if layers else prev_op.weight.device
        scales = scales.to(device=device, dtype=layers[0].weight.dtype)

        if isinstance(prev_op, nn.Linear):
            if len(layers) != 1:
                raise ValueError("Linear-to-linear MBQ scaling expects a single target layer.")
            _scale_fc_fc(prev_op, layers[0], scales)
        elif _is_norm_module(prev_op):
            _scale_norm_linears(prev_op, layers, scales)
        else:
            raise NotImplementedError(f"Unsupported MBQ previous op: {type(prev_op)}")

        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(_reshape_scales_for_input(scales.to(inp.device), inp))


def _prefix_scales(layer_idx, scales_list):
    prefixed = []
    prefix = f"layers.{layer_idx}."
    for prev_op_name, layer_names, scales in scales_list:
        prefixed.append(
            (
                prefix + prev_op_name,
                tuple(prefix + layer_name for layer_name in layer_names),
                scales.cpu(),
            )
        )
    return prefixed


@torch.no_grad()
def _apply_mbq_scales(backbone_model, scales_list):
    _apply_scales_to_module(backbone_model, scales_list, input_feat_dict=None)


class GradCacheHook:
    def __init__(self, vis_masks, cap_masks):
        if vis_masks is None or cap_masks is None:
            raise ValueError("MBQ reweight requires both vision_mask and caption_mask.")
        self.hooks = []
        self.vis_masks = vis_masks.cpu()
        self.cap_masks = cap_masks.cpu()
        self.steps = {}
        self.grad_dict = {}

    def cache_grad_hook(self, _module, _grad_input, grad_output, name):
        if name not in self.steps:
            self.steps[name] = 0
        if name not in self.grad_dict:
            self.grad_dict[name] = {"vis_grad": [], "cap_grad": []}

        output_grad = grad_output[0].float()
        step = self.steps[name]
        batch_size = output_grad.shape[0]

        for batch_idx in range(batch_size):
            vis_mask = self.vis_masks[step].to(output_grad.device)
            cap_mask = self.cap_masks[step].to(output_grad.device)

            vis_grad_avg = output_grad[batch_idx][vis_mask].abs().mean()
            cap_grad_avg = output_grad[batch_idx][cap_mask].abs().mean()

            self.grad_dict[name]["vis_grad"].append(vis_grad_avg.detach().cpu())
            self.grad_dict[name]["cap_grad"].append(cap_grad_avg.detach().cpu())
            step += 1

        self.steps[name] = step

    def register_hooks(self, layers):
        target_keywords = (
            "o_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        for name, module in layers.named_modules():
            if isinstance(module, nn.Linear) and any(keyword in name for keyword in target_keywords):
                self.hooks.append(
                    module.register_full_backward_hook(
                        lambda m, gi, go, n=name: self.cache_grad_hook(
                            m,
                            gi,
                            go,
                            f"layers.{n}",
                        )
                    )
                )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_avg_grad_dict(self):
        avg_grad_dict = {}
        for name, grad_values in self.grad_dict.items():
            avg_grad_dict[name] = {
                "vis_avg_grad": torch.mean(torch.stack(grad_values["vis_grad"])).item(),
                "cap_avg_grad": torch.mean(torch.stack(grad_values["cap_grad"])).item(),
            }
        return avg_grad_dict


def _build_model_inputs(multimodal_inputs):
    prompt_inputs = multimodal_inputs.get("prompt_inputs")
    prompt_kwargs = multimodal_inputs.get("prompt_kwargs")
    if prompt_inputs is None or prompt_kwargs is None:
        raise ValueError(
            "MBQ requires calibration_mode='mbq' so that prompt_inputs/prompt_kwargs are available."
        )

    model_inputs = {
        "inputs_embeds": prompt_inputs["inputs_embeds"],
        "use_cache": False,
    }
    for key in ("attention_mask", "position_ids", "labels"):
        if key in prompt_kwargs and prompt_kwargs[key] is not None:
            model_inputs[key] = prompt_kwargs[key]

    return (
        model_inputs,
        prompt_kwargs.get("vision_mask"),
        prompt_kwargs.get("caption_mask"),
    )


def _collect_reweight_statistics(
    lisa_model,
    model_inputs,
    *,
    vision_mask,
    caption_mask,
    runtime_device,
    runtime_dtype,
):
    grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)
    layers = _get_decoder_layers(lisa_model.get_model())

    lisa_model.to(device=runtime_device, dtype=runtime_dtype)
    grad_cache.register_hooks(layers)
    lisa_model.zero_grad(set_to_none=True)

    mini_batch = 1
    total_samples = model_inputs["inputs_embeds"].shape[0]
    accum_steps = max(total_samples // mini_batch, 1)

    try:
        with torch.enable_grad():
            for start_idx in tqdm.tqdm(
                range(0, total_samples, mini_batch),
                desc="Running MBQ gradient reweight",
            ):
                mini_inputs = {}
                for key, value in model_inputs.items():
                    if not isinstance(value, torch.Tensor):
                        mini_inputs[key] = value
                        continue
                    sliced_value = value[start_idx : start_idx + mini_batch]
                    mini_inputs[key] = _to_runtime_tensor(
                        sliced_value,
                        runtime_device,
                        runtime_dtype,
                    )

                outputs = LlavaLlamaForCausalLM.forward(lisa_model, **mini_inputs)
                loss = outputs[0] / accum_steps
                loss.backward()
    finally:
        grad_avg_dict = grad_cache.get_avg_grad_dict()
        grad_cache.remove_hooks()
        lisa_model.zero_grad(set_to_none=True)
        lisa_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attn_list = []
    mlp_list = []
    for key_name, values in grad_avg_dict.items():
        denominator = max(values["cap_avg_grad"], 1e-6)
        ratio = values["vis_avg_grad"] / denominator
        if "down_proj" in key_name:
            mlp_list.append(ratio)
        if "o_proj" in key_name:
            attn_list.append(ratio)

    attn_median = float(torch.tensor(attn_list).median().item()) if attn_list else 1.0
    mlp_median = float(torch.tensor(mlp_list).median().item()) if mlp_list else 1.0
    return grad_avg_dict, attn_median, mlp_median


def _build_reweight_ratio_dict(
    layer_idx,
    grad_avg_dict,
    *,
    attn_median,
    mlp_median,
):
    ratio_dict = {"attn": None, "mlp": None}
    prefix = f"layers.{layer_idx}."

    for key_name, values in grad_avg_dict.items():
        if not key_name.startswith(prefix):
            continue

        denominator = max(values["cap_avg_grad"], 1e-6)
        ratio = values["vis_avg_grad"] / denominator
        if "o_proj" in key_name:
            ratio_dict["attn"] = max(ratio, attn_median)
        elif "down_proj" in key_name:
            ratio_dict["mlp"] = max(ratio, mlp_median)

    return ratio_dict


def _build_quantized_layer_copy(
    layer,
    *,
    wa_quant,
    w_bit,
    a_bit,
    q_config,
):
    layer_q = copy.deepcopy(layer)
    named_linears = _get_named_linears(layer_q)

    for name, module in named_linears.items():
        if wa_quant:
            quantized_linear = PseudoQuantLinear.from_float(
                module,
                weight_quant="per_channel",
                act_quant="per_token",
                w_bit=w_bit,
                a_bit=a_bit,
                act_quant_mode="always",
            )
            _set_submodule(layer_q, name, quantized_linear)
        else:
            module.weight.data = pseudo_quantize_tensor(
                module.weight.data,
                n_bits=w_bit,
                zero_point=q_config.get("zero_point", True),
                q_group_size=q_config.get("q_group_size", 128),
            )

    return layer_q


@torch.no_grad()
def run_mbq_search(
    lisa_model,
    multimodal_inputs,
    *,
    w_bit,
    a_bit,
    q_config,
    wa_quant=False,
    reweight=False,
    distort=False,
    loss_mode="mae",
    runtime_device=None,
    runtime_dtype=None,
):
    runtime_device = _get_runtime_device(runtime_device)
    if runtime_dtype is None:
        runtime_dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32

    model_inputs, vision_mask, caption_mask = _build_model_inputs(multimodal_inputs)
    inps, layer_kwargs = _capture_first_layer_inputs(
        lisa_model,
        model_inputs,
        runtime_device,
        runtime_dtype,
    )

    grad_avg_dict = None
    attn_median = 1.0
    mlp_median = 1.0
    if reweight:
        grad_avg_dict, attn_median, mlp_median = _collect_reweight_statistics(
            lisa_model,
            model_inputs,
            vision_mask=vision_mask,
            caption_mask=caption_mask,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
        )

    inps_distort = inps.clone() if distort else None
    layers = _get_decoder_layers(lisa_model.get_model())
    mbq_results = {"scale": []}

    for layer_idx in tqdm.tqdm(range(len(layers)), desc="Running MBQ"):
        layer = layers[layer_idx].to(device=runtime_device, dtype=runtime_dtype)
        current_inps = inps.to(device=runtime_device, dtype=runtime_dtype)
        current_kwargs = _move_kwargs_to_device(layer_kwargs, runtime_device, runtime_dtype)

        next_inps, input_feat = _cache_layer_inputs(layer, current_inps, current_kwargs)

        if reweight and grad_avg_dict is not None:
            reweight_ratio_dict = _build_reweight_ratio_dict(
                layer_idx,
                grad_avg_dict,
                attn_median=attn_median,
                mlp_median=mlp_median,
            )
            ans_mask = caption_mask
            vis_mask = vision_mask
        else:
            reweight_ratio_dict = {"attn": None, "mlp": None}
            ans_mask = None
            vis_mask = None

        local_scales = _search_llama_layer_scales(
            layer,
            current_kwargs,
            input_feat,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            wa_quant=wa_quant,
            reweight_ratio_dict=reweight_ratio_dict,
            ans_mask=ans_mask,
            vis_mask=vis_mask,
            loss_mode=loss_mode,
        )
        _apply_scales_to_module(layer, local_scales, input_feat_dict=input_feat)
        mbq_results["scale"].extend(_prefix_scales(layer_idx, local_scales))

        if distort:
            layer_q = _build_quantized_layer_copy(
                layer,
                wa_quant=wa_quant,
                w_bit=w_bit,
                a_bit=a_bit,
                q_config=q_config,
            ).to(device=runtime_device, dtype=runtime_dtype)
            distort_inputs = inps_distort.to(device=runtime_device, dtype=runtime_dtype)
            inps_distort = layer_q(distort_inputs, **current_kwargs)[0].detach().cpu()
            del layer_q, distort_inputs

        inps = next_inps.detach().cpu()
        layers[layer_idx] = layer.cpu()

        del current_inps, current_kwargs, next_inps, input_feat, local_scales
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return mbq_results


@torch.no_grad()
def _pseudo_quantize_backbone_weight_only(backbone_model, *, w_bit, q_config):
    layers = _get_decoder_layers(backbone_model)
    for layer in layers:
        for module_name in QUANTIZED_LINEAR_NAMES:
            module = _get_submodule(layer, module_name)
            if not isinstance(module, nn.Linear):
                continue
            module.weight.data = pseudo_quantize_tensor(
                module.weight.data,
                n_bits=w_bit,
                zero_point=q_config.get("zero_point", True),
                q_group_size=q_config.get("q_group_size", 128),
            )


@torch.no_grad()
def _pseudo_quantize_backbone_weight_act(
    backbone_model,
    *,
    w_bit,
    a_bit,
    act_quant_mode="prefill",
):
    layers = _get_decoder_layers(backbone_model)
    for layer in layers:
        for module_name in QUANTIZED_LINEAR_NAMES:
            module = _get_submodule(layer, module_name)
            if not isinstance(module, nn.Linear):
                continue
            quantized_linear = PseudoQuantLinear.from_float(
                module,
                weight_quant="per_channel",
                act_quant="per_token",
                w_bit=w_bit,
                a_bit=a_bit,
                act_quant_mode=act_quant_mode,
            )
            _set_submodule(layer, module_name, quantized_linear)


def ensure_mbq_scales(
    model_path,
    quant_kwargs,
    *,
    force=False,
    runtime_device=None,
    runtime_dtype=None,
):
    patch_transformers_compat()

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs for MBQ must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("MBQ quant_kwargs must include '_config_path'.")

    config = load_mbq_config(config_path)
    scale_path = Path(
        config.get(
            "mbq_scale_path",
            _default_scale_path(
                model_path,
                wa_quant=config.get("wa_quant", False),
                w_bit=config.get("w_bit", 4),
                a_bit=config.get("a_bit", 8),
            ),
        )
    )
    if scale_path.exists() and not force:
        return str(scale_path)

    tokenizer, calibration_records, multimodal_inputs = build_calibration_data(str(model_path), config)
    if not calibration_records:
        raise ValueError("Calibration dataset is empty.")

    if runtime_dtype is None:
        runtime_device = _get_runtime_device(runtime_device)
        runtime_dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32

    lisa_model = build_lisa_model(model_path, config, tokenizer)
    try:
        mbq_results = run_mbq_search(
            lisa_model,
            multimodal_inputs,
            w_bit=config.get("w_bit", 4),
            a_bit=config.get("a_bit", 8),
            q_config={
                "zero_point": config.get("zero_point", True),
                "q_group_size": config.get("q_group_size", 128),
            },
            wa_quant=config.get("wa_quant", False),
            reweight=config.get("reweight", False),
            distort=config.get("distort", False),
            loss_mode=config.get("loss_mode", "mae"),
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
        )
        _save_mbq_results(scale_path, model_path, config, mbq_results)
    finally:
        del lisa_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return str(scale_path)


def load_mbq_backbone_into_lisa(
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
        raise ValueError("quant_kwargs for MBQ must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("MBQ quant_kwargs must include '_config_path'.")

    config = load_mbq_config(config_path)
    scale_path = ensure_mbq_scales(
        model_path,
        quant_kwargs,
        force=force,
        runtime_device=device,
        runtime_dtype=torch_dtype,
    )
    mbq_results = _load_mbq_results(scale_path)

    backbone_model = lisa_model.get_model()
    _apply_mbq_scales(backbone_model, mbq_results["scale"])

    if config.get("wa_quant", False):
        _pseudo_quantize_backbone_weight_act(
            backbone_model,
            w_bit=config.get("w_bit", 4),
            a_bit=config.get("a_bit", 8),
            act_quant_mode=config.get("activation_quant_mode", "prefill"),
        )
    else:
        _pseudo_quantize_backbone_weight_only(
            backbone_model,
            w_bit=config.get("w_bit", 4),
            q_config={
                "zero_point": config.get("zero_point", True),
                "q_group_size": config.get("q_group_size", 128),
            },
        )

    lisa_model.quantization_method = "mbq"
    lisa_model.config.quantization_config = {
        "quant_method": "mbq",
        "w_bit": config.get("w_bit", 4),
        "a_bit": config.get("a_bit", 8),
        "wa_quant": bool(config.get("wa_quant", False)),
        "zero_point": bool(config.get("zero_point", True)),
        "q_group_size": config.get("q_group_size", 128),
        "loss_mode": config.get("loss_mode", "mae"),
        "reweight": bool(config.get("reweight", False)),
        "distort": bool(config.get("distort", False)),
        "activation_quant_mode": config.get("activation_quant_mode", "prefill"),
    }
    return lisa_model


def main():
    args = parse_args()
    config = load_mbq_config(args.config)
    scale_path = ensure_mbq_scales(
        model_path=config["model_path"],
        quant_kwargs={"_config_path": config["_config_path"]},
        force=args.force,
    )
    print(scale_path)


if __name__ == "__main__":
    main()
