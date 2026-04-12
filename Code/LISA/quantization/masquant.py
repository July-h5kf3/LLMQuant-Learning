import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    get_decoder_layers,
    get_submodule_by_name,
    load_method_quant_config,
    patch_transformers_compat,
    set_submodule_by_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MASQuant activation scales for the LISA LLM backbone."
    )
    parser.add_argument(
        "--config",
        default="configs/quant/masquant.yaml",
        type=str,
        help="Path to the MASQuant YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate MASQuant scales even if they already exist.",
    )
    return parser.parse_args()


def load_masquant_config(config_path):
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=(
            "model_path",
            "masquant_scale_path",
            "dataset_dir",
            "vision_tower",
            "vision_pretrained",
        ),
        bool_defaults={
            "keep_lm_export": False,
            "wa_quant": False,
            "cmc": False,
            "quant_cmc": False,
            "zero_point": False,
        },
    )


def _default_scale_path(model_path, *, scale_mode, wa_quant, cmc, w_bit, a_bit):
    model_path = Path(model_path)
    quant_suffix = f"w{w_bit}a{a_bit}" if wa_quant else f"w{w_bit}"
    cmc_suffix = "_cmc" if cmc else ""
    return model_path.parent / f"{model_path.name}_masquant_{scale_mode}{cmc_suffix}_{quant_suffix}.pt"


def _load_masquant_payload(scale_path):
    payload = torch.load(scale_path, map_location="cpu")
    if isinstance(payload, dict) and "act_scales" in payload:
        act_scales = payload["act_scales"]
        cmc_adapters = payload.get("cmc_adapters", {})
    else:
        act_scales = payload
        cmc_adapters = {}

    if not isinstance(act_scales, dict) or not act_scales:
        raise ValueError(f"Invalid MASQuant scale payload: {scale_path}")

    act_scales = {
        key: value.cpu() if torch.is_tensor(value) else torch.as_tensor(value).cpu()
        for key, value in act_scales.items()
    }
    return {
        "act_scales": act_scales,
        "cmc_adapters": cmc_adapters or {},
    }


def _save_masquant_payload(scale_path, model_path, config, act_scales, cmc_adapters=None):
    scale_path = Path(scale_path)
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "act_scales": act_scales,
            "cmc_adapters": cmc_adapters or {},
            "meta": {
                "source_model_path": str(model_path),
                "scale_path": str(scale_path),
                "config_path": config["_config_path"],
                "scale_mode": config.get("scale_mode", "split"),
                "alpha": config.get("alpha", 0.5),
                "w_bit": config.get("w_bit", 4),
                "a_bit": config.get("a_bit", 8),
                "wa_quant": bool(config.get("wa_quant", False)),
                "cmc": bool(config.get("cmc", False)),
                "cmc_rank": config.get("cmc_rank", 0.05),
                "quant_cmc": bool(config.get("quant_cmc", False)),
                "num_scale_tensors": len(act_scales),
                "num_cmc_adapters": len(cmc_adapters or {}),
            },
        },
        scale_path,
    )


def _get_runtime_device(runtime_device):
    if runtime_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(runtime_device))


def _to_runtime_tensor(value, device, runtime_dtype):
    if value is None:
        return None
    if value.dtype.is_floating_point:
        return value.to(device=device, dtype=runtime_dtype)
    return value.to(device=device)


def _iter_prompt_batches(multimodal_inputs, batch_size):
    prompt_inputs = multimodal_inputs.get("prompt_inputs")
    prompt_kwargs = multimodal_inputs.get("prompt_kwargs")
    if prompt_inputs is None or prompt_kwargs is None:
        raise ValueError("MASQuant requires calibration_mode='mbq'.")

    inputs_embeds = prompt_inputs["inputs_embeds"]
    total_samples = inputs_embeds.shape[0]
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch = {"inputs_embeds": inputs_embeds[start:end]}
        for key in ("attention_mask", "position_ids", "vision_mask"):
            value = prompt_kwargs.get(key)
            if value is not None:
                batch[key] = value[start:end]
        yield batch


def _masked_absmax(hidden_states, token_mask):
    hidden_dim = hidden_states.shape[-1]
    flattened = hidden_states.reshape(-1, hidden_dim).abs()
    flattened_mask = token_mask.reshape(-1)
    if not torch.any(flattened_mask):
        return torch.zeros(hidden_dim, dtype=torch.float32, device="cpu")
    return flattened[flattened_mask].amax(dim=0).float().cpu()


def _update_scale_dict(act_scales, name, hidden_states, attention_mask, vision_mask):
    valid_mask = attention_mask.to(torch.bool)
    if valid_mask.shape != hidden_states.shape[:2]:
        raise ValueError("MASQuant attention mask must match hidden-state tokens.")

    vision_mask = vision_mask.to(torch.bool) & valid_mask
    text_mask = valid_mask & ~vision_mask

    incoming = {
        f"{name}.all_in_one_scale": _masked_absmax(hidden_states, valid_mask),
        f"{name}.text_scale": _masked_absmax(hidden_states, text_mask),
        f"{name}.vision_scale": _masked_absmax(hidden_states, vision_mask),
    }

    for key, value in incoming.items():
        if key in act_scales:
            act_scales[key] = torch.maximum(act_scales[key], value)
        else:
            act_scales[key] = value


@torch.no_grad()
def collect_masquant_act_scales(
    lisa_model,
    multimodal_inputs,
    *,
    runtime_device=None,
    runtime_dtype=None,
    calibration_batch_size=1,
):
    device = _get_runtime_device(runtime_device)
    if runtime_dtype is None:
        runtime_dtype = torch.float16 if device.type == "cuda" else torch.float32

    act_scales = {}
    current_masks = {}
    lisa_model.eval()
    lisa_model.to(device=device, dtype=runtime_dtype)
    module_root = lisa_model.get_model()

    def stat_input_hook(_module, inputs, _output, name):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        _update_scale_dict(
            act_scales,
            name,
            hidden_states.detach(),
            current_masks["attention_mask"],
            current_masks["vision_mask"],
        )

    hooks = []
    for name, module in _iter_target_linears(module_root):
        hooks.append(
            module.register_forward_hook(
                lambda m, x, y, n=name: stat_input_hook(m, x, y, n)
            )
        )

    try:
        for batch in tqdm.tqdm(
            _iter_prompt_batches(multimodal_inputs, calibration_batch_size),
            desc="Collecting MASQuant scales",
        ):
            attention_mask = batch["attention_mask"].to(device=device)
            vision_mask = batch["vision_mask"].to(device=device)
            current_masks["attention_mask"] = attention_mask
            current_masks["vision_mask"] = vision_mask

            model_inputs = {
                "inputs_embeds": batch["inputs_embeds"].to(
                    device=device,
                    dtype=runtime_dtype,
                ),
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if "position_ids" in batch:
                model_inputs["position_ids"] = batch["position_ids"].to(device=device)

            LlavaLlamaForCausalLM.forward(lisa_model, **model_inputs)
    finally:
        for hook in hooks:
            hook.remove()
        lisa_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return act_scales


def _share_scale_source(module_name):
    if module_name in {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}:
        return "self_attn.q_proj"
    if module_name in {"mlp.gate_proj", "mlp.up_proj"}:
        return "mlp.up_proj"
    return module_name


def _smooth_scale_from_act(act_scale, module, alpha):
    act_scale = act_scale.to(device=module.weight.device, dtype=torch.float32).clamp(min=1e-5)
    weight_scale = module.weight.detach().abs().amax(dim=0).float().clamp(min=1e-5)
    return (act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)).clamp(min=1e-5)


def _resolve_module_scales(act_scales, layer_idx, module_name, module, *, alpha, scale_mode):
    if module_name == "mlp.down_proj":
        ones = torch.ones(module.in_features, dtype=torch.float32)
        return ones, None

    source_name = _share_scale_source(module_name)
    key_prefix = f"layers.{layer_idx}.{source_name}"

    if scale_mode == "merged":
        text_scale = _smooth_scale_from_act(
            act_scales[f"{key_prefix}.all_in_one_scale"],
            module,
            alpha,
        )
        return text_scale, None

    if scale_mode != "split":
        raise ValueError("MASQuant scale_mode must be 'split' or 'merged'.")

    text_scale = _smooth_scale_from_act(
        act_scales[f"{key_prefix}.text_scale"],
        module,
        alpha,
    )
    vision_scale = _smooth_scale_from_act(
        act_scales[f"{key_prefix}.vision_scale"],
        module,
        alpha,
    )
    return text_scale, vision_scale


def _parse_decoder_linear_name(full_name):
    parts = full_name.split(".")
    if len(parts) < 4 or parts[0] != "layers":
        raise ValueError(f"Unexpected MASQuant linear name: {full_name}")
    return int(parts[1]), ".".join(parts[2:])


def _iter_target_linears(module_root):
    for name, module in module_root.named_modules():
        if not name.startswith("layers."):
            continue
        if isinstance(module, nn.Linear) and any(
            name.endswith(module_name) for module_name in QUANTIZED_LINEAR_NAMES
        ):
            yield name, module


def _is_cmc_source_name(module_name):
    return module_name in {
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
    }


def _is_cmc_target_name(module_name):
    return module_name != "mlp.down_proj"


def _resolve_full_name_scales(act_scales, full_name, module, *, alpha, scale_mode):
    layer_idx, module_name = _parse_decoder_linear_name(full_name)
    return _resolve_module_scales(
        act_scales,
        layer_idx,
        module_name,
        module,
        alpha=alpha,
        scale_mode=scale_mode,
    )


@torch.no_grad()
def collect_masquant_vision_covs(
    lisa_model,
    multimodal_inputs,
    act_scales,
    *,
    alpha=0.5,
    scale_mode="split",
    runtime_device=None,
    runtime_dtype=None,
    calibration_batch_size=1,
):
    if scale_mode != "split":
        return {}

    device = _get_runtime_device(runtime_device)
    if runtime_dtype is None:
        runtime_dtype = torch.float16 if device.type == "cuda" else torch.float32

    covs = {}
    scale_lookup = {}
    current_masks = {}
    lisa_model.eval()
    lisa_model.to(device=device, dtype=runtime_dtype)
    module_root = lisa_model.get_model()

    for name, module in _iter_target_linears(module_root):
        _, module_name = _parse_decoder_linear_name(name)
        if not _is_cmc_source_name(module_name):
            continue
        _, vision_scale = _resolve_full_name_scales(
            act_scales,
            name,
            module,
            alpha=alpha,
            scale_mode=scale_mode,
        )
        if vision_scale is not None:
            scale_lookup[name] = vision_scale.cpu()

    def cov_input_hook(_module, inputs, _output, name):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        attention_mask = current_masks["attention_mask"].to(torch.bool)
        vision_mask = current_masks["vision_mask"].to(torch.bool) & attention_mask
        if not torch.any(vision_mask):
            return

        scale = scale_lookup[name].to(device=hidden_states.device, dtype=hidden_states.dtype)
        scaled_hidden = hidden_states.detach() / scale.view(1, 1, -1)
        vision_hidden = scaled_hidden[vision_mask].float()
        incoming_cov = vision_hidden.transpose(0, 1).matmul(vision_hidden).cpu()
        if name in covs:
            covs[name].add_(incoming_cov)
        else:
            covs[name] = incoming_cov

    hooks = []
    for name, module in _iter_target_linears(module_root):
        if name in scale_lookup:
            hooks.append(
                module.register_forward_hook(
                    lambda m, x, y, n=name: cov_input_hook(m, x, y, n)
                )
            )

    try:
        for batch in tqdm.tqdm(
            _iter_prompt_batches(multimodal_inputs, calibration_batch_size),
            desc="Collecting MASQuant CMC covariances",
        ):
            attention_mask = batch["attention_mask"].to(device=device)
            vision_mask = batch["vision_mask"].to(device=device)
            current_masks["attention_mask"] = attention_mask
            current_masks["vision_mask"] = vision_mask

            model_inputs = {
                "inputs_embeds": batch["inputs_embeds"].to(
                    device=device,
                    dtype=runtime_dtype,
                ),
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if "position_ids" in batch:
                model_inputs["position_ids"] = batch["position_ids"].to(device=device)

            LlavaLlamaForCausalLM.forward(lisa_model, **model_inputs)
    finally:
        for hook in hooks:
            hook.remove()
        lisa_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return covs


def _target_cmc_rank(rank, min_dim):
    rank = float(rank)
    if rank <= 0:
        return 0
    if rank <= 1:
        rank = int(min_dim * rank)
    else:
        rank = int(rank)
    return max(1, min(rank, min_dim))


def _build_whitening_matrix(cov, *, device):
    cov = cov.to(device=device, dtype=torch.float32)
    jitter = cov.diagonal().mean().clamp(min=1e-6) * 1e-4
    eye = torch.eye(cov.shape[0], device=device, dtype=cov.dtype)
    return torch.linalg.cholesky(cov + jitter * eye).transpose(0, 1)


def _low_rank_cmc_adapter(
    module,
    text_scale,
    vision_scale,
    cov,
    *,
    w_bit,
    weight_zero_point,
    q_group_size,
    rank,
    oversample,
    niter,
    quant_cmc,
    device,
):
    rank = _target_cmc_rank(rank, min(module.weight.shape))
    if rank <= 0:
        return None

    module = module.to(device)
    text_scale = text_scale.to(device=device, dtype=module.weight.dtype)
    vision_scale = vision_scale.to(device=device, dtype=module.weight.dtype)

    text_weight = MASQuantLinear._quantize_scaled_weight(
        module,
        text_scale,
        w_bit=w_bit,
        zero_point=weight_zero_point,
        q_group_size=q_group_size,
    ).float()
    vision_weight = module.weight.data * vision_scale.view(1, -1)
    if quant_cmc:
        vision_weight = pseudo_quantize_tensor(
            vision_weight,
            n_bits=w_bit,
            zero_point=weight_zero_point,
            q_group_size=q_group_size,
        )
    vision_weight = vision_weight.float()

    residual = (vision_weight - text_weight).transpose(0, 1).contiguous()
    whitening = _build_whitening_matrix(cov, device=device)
    weighted_residual = whitening.matmul(residual)

    q_rank = min(rank + int(oversample), min(weighted_residual.shape))
    u, singular_values, v = torch.svd_lowrank(
        weighted_residual,
        q=q_rank,
        niter=int(niter),
    )
    u = u[:, :rank]
    singular_values = singular_values[:rank]
    v = v[:, :rank]

    left = torch.linalg.solve_triangular(whitening, u, upper=True)
    right = singular_values.view(-1, 1) * v.transpose(0, 1)
    return {
        "L": left.detach().cpu().to(torch.float16),
        "R": right.detach().cpu().to(torch.float16),
        "rank": rank,
    }


@torch.no_grad()
def build_masquant_cmc_adapters(
    lisa_model,
    multimodal_inputs,
    act_scales,
    *,
    alpha=0.5,
    scale_mode="split",
    w_bit=4,
    weight_zero_point=False,
    q_group_size=-1,
    rank=0.05,
    oversample=32,
    niter=2,
    quant_cmc=False,
    runtime_device=None,
    runtime_dtype=None,
    calibration_batch_size=1,
):
    if scale_mode != "split":
        return {}

    device = _get_runtime_device(runtime_device)
    if runtime_dtype is None:
        runtime_dtype = torch.float16 if device.type == "cuda" else torch.float32

    covs = collect_masquant_vision_covs(
        lisa_model,
        multimodal_inputs,
        act_scales,
        alpha=alpha,
        scale_mode=scale_mode,
        runtime_device=device,
        runtime_dtype=runtime_dtype,
        calibration_batch_size=calibration_batch_size,
    )

    module_root = lisa_model.get_model()
    cmc_adapters = {}
    for name, module in tqdm.tqdm(
        list(_iter_target_linears(module_root)),
        desc="Building MASQuant CMC adapters",
    ):
        layer_idx, module_name = _parse_decoder_linear_name(name)
        if not _is_cmc_target_name(module_name):
            continue

        source_module_name = _share_scale_source(module_name)
        source_name = f"layers.{layer_idx}.{source_module_name}"
        cov = covs.get(source_name)
        if cov is None:
            continue

        text_scale, vision_scale = _resolve_full_name_scales(
            act_scales,
            name,
            module,
            alpha=alpha,
            scale_mode=scale_mode,
        )
        if vision_scale is None:
            continue

        adapter = _low_rank_cmc_adapter(
            module,
            text_scale,
            vision_scale,
            cov,
            w_bit=w_bit,
            weight_zero_point=weight_zero_point,
            q_group_size=q_group_size,
            rank=rank,
            oversample=oversample,
            niter=niter,
            quant_cmc=quant_cmc,
            device=device,
        )
        if adapter is not None:
            cmc_adapters[name] = adapter

        module.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del covs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cmc_adapters


class MASQuantLinear(PseudoQuantLinear):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=True,
        text_scale,
        vision_scale=None,
        wa_quant=False,
        a_bit=8,
        w_bit=4,
        weight_zero_point=False,
        q_group_size=-1,
        cmc_adapter=None,
        act_quant_mode="prefill",
        dtype=torch.float32,
        device=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            act_quant="per_token",
            a_bit=a_bit,
            w_bit=w_bit,
            act_quant_mode=act_quant_mode,
            dtype=dtype,
            device=device,
        )
        self.wa_quant = wa_quant
        self.weight_zero_point = weight_zero_point
        self.q_group_size = q_group_size
        self.current_vision_mask = None
        self.scale_mode = "split" if vision_scale is not None else "merged"
        self.weight_quant_name = "masquant"
        self.register_buffer("text_smooth_scale", text_scale.to(device=device, dtype=dtype))
        self.register_buffer(
            "vision_smooth_scale",
            None if vision_scale is None else vision_scale.to(device=device, dtype=dtype),
        )
        if cmc_adapter is None:
            self.register_buffer("cmc_l", None)
            self.register_buffer("cmc_r", None)
        else:
            self.register_buffer("cmc_l", cmc_adapter["L"].to(device=device, dtype=dtype))
            self.register_buffer("cmc_r", cmc_adapter["R"].to(device=device, dtype=dtype))

    @staticmethod
    def _quantize_scaled_weight(module, scale, *, w_bit, zero_point, q_group_size):
        scale = scale.to(device=module.weight.device, dtype=module.weight.dtype)
        scaled_weight = module.weight.data * scale.view(1, -1)
        return pseudo_quantize_tensor(
            scaled_weight,
            n_bits=w_bit,
            zero_point=zero_point,
            q_group_size=q_group_size,
        )

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        module,
        *,
        text_scale,
        vision_scale=None,
        wa_quant=False,
        w_bit=4,
        a_bit=8,
        weight_zero_point=False,
        q_group_size=-1,
        cmc_adapter=None,
        act_quant_mode="prefill",
    ):
        new_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            text_scale=text_scale,
            vision_scale=vision_scale,
            wa_quant=wa_quant,
            a_bit=a_bit,
            w_bit=w_bit,
            weight_zero_point=weight_zero_point,
            q_group_size=q_group_size,
            cmc_adapter=cmc_adapter,
            act_quant_mode=act_quant_mode,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        new_module.weight.copy_(
            cls._quantize_scaled_weight(
                module,
                text_scale,
                w_bit=w_bit,
                zero_point=weight_zero_point,
                q_group_size=q_group_size,
            )
        )
        if vision_scale is not None:
            # MASQuant shares one text-smoothed quantized weight across modalities.
            # Modality-specific scales only rescale the corresponding activations.
            new_module.vision_smooth_scale.copy_(
                vision_scale.to(
                    device=new_module.vision_smooth_scale.device,
                    dtype=new_module.vision_smooth_scale.dtype,
                )
            )
        if module.bias is not None:
            new_module.bias.copy_(module.bias.data)
        return new_module

    def set_vision_mask(self, vision_mask):
        self.current_vision_mask = vision_mask

    def _reshape_scale(self, scale, x):
        shape = [1] * x.ndim
        shape[-1] = scale.numel()
        return scale.to(device=x.device, dtype=x.dtype).view(*shape)

    def _maybe_quantize_activation(self, x):
        if self.wa_quant and self.a_bit < 16 and self.should_quantize_activation():
            return self.act_quant(x)
        return x

    def _linear(self, x, weight, bias):
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    @torch.no_grad()
    def forward(self, x):
        vision_mask = self.current_vision_mask
        use_split = (
            self.vision_smooth_scale is not None
            and self.activation_stage == "prefill"
            and vision_mask is not None
            and tuple(vision_mask.shape) == tuple(x.shape[:2])
        )
        if not use_split:
            text_scale = self._reshape_scale(self.text_smooth_scale, x)
            quantized_text = self._maybe_quantize_activation(x / text_scale)
            return self._linear(quantized_text, self.weight, self.bias)

        vision_mask = vision_mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1)
        text_scale = self._reshape_scale(self.text_smooth_scale, x)
        text_x = x.masked_fill(vision_mask, 0)
        vision_x = x.masked_fill(~vision_mask, 0)
        text_x = self._maybe_quantize_activation(text_x / text_scale)

        vision_scale = self._reshape_scale(self.vision_smooth_scale, x)
        vision_x = vision_x / vision_scale
        quantized_vision_x = self._maybe_quantize_activation(vision_x)

        out = self._linear(text_x, self.weight, self.bias)
        out = out + self._linear(quantized_vision_x, self.weight, None)
        if self.cmc_l is not None and self.cmc_r is not None:
            cmc_l = self.cmc_l.to(device=x.device, dtype=x.dtype)
            cmc_r = self.cmc_r.to(device=x.device, dtype=x.dtype)
            out = out + vision_x.matmul(cmc_l).matmul(cmc_r)
        return out

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, w_bit={self.w_bit}, a_bit={self.a_bit}, "
            f"wa_quant={self.wa_quant}, scale_mode="
            f"{self.scale_mode}, cmc={self.cmc_l is not None}, "
            f"act_quant_mode={self.act_quant_mode}, activation_stage={self.activation_stage}"
        )


@torch.no_grad()
def pseudo_quantize_masquant_backbone(
    backbone_model,
    act_scales,
    *,
    alpha=0.5,
    scale_mode="split",
    wa_quant=False,
    w_bit=4,
    a_bit=8,
    weight_zero_point=False,
    q_group_size=-1,
    cmc_adapters=None,
    act_quant_mode="prefill",
):
    if cmc_adapters is None:
        cmc_adapters = {}

    layers = get_decoder_layers(backbone_model)
    for layer_idx, layer in enumerate(layers):
        for module_name in QUANTIZED_LINEAR_NAMES:
            module = get_submodule_by_name(layer, module_name)
            if not isinstance(module, nn.Linear):
                continue

            text_scale, vision_scale = _resolve_module_scales(
                act_scales,
                layer_idx,
                module_name,
                module,
                alpha=alpha,
                scale_mode=scale_mode,
            )
            set_submodule_by_name(
                layer,
                module_name,
                MASQuantLinear.from_float(
                    module,
                    text_scale=text_scale,
                    vision_scale=vision_scale,
                    wa_quant=wa_quant,
                    w_bit=w_bit,
                    a_bit=a_bit,
                    weight_zero_point=weight_zero_point,
                    q_group_size=q_group_size,
                    cmc_adapter=cmc_adapters.get(f"layers.{layer_idx}.{module_name}"),
                    act_quant_mode=act_quant_mode,
                ),
            )


def ensure_masquant_scales(
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
        raise ValueError("quant_kwargs for MASQuant must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("MASQuant quant_kwargs must include '_config_path'.")

    config = load_masquant_config(config_path)
    scale_mode = config.get("scale_mode", "split")
    scale_path = Path(
        config.get(
            "masquant_scale_path",
            _default_scale_path(
                model_path,
                scale_mode=scale_mode,
                wa_quant=config.get("wa_quant", False),
                cmc=config.get("cmc", False),
                w_bit=config.get("w_bit", 4),
                a_bit=config.get("a_bit", 8),
            ),
        )
    )
    if scale_path.exists() and not force:
        payload = _load_masquant_payload(scale_path)
        if not config.get("cmc", False) or payload["cmc_adapters"]:
            return str(scale_path)

    tokenizer, calibration_records, multimodal_inputs = build_calibration_data(str(model_path), config)
    if not calibration_records:
        raise ValueError("Calibration dataset is empty.")

    owns_model = lisa_model is None
    try:
        if lisa_model is None:
            lisa_model = build_lisa_model(model_path, config, tokenizer)
        act_scales = collect_masquant_act_scales(
            lisa_model,
            multimodal_inputs,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            calibration_batch_size=config.get("calibration_batch_size", 1),
        )
        cmc_adapters = {}
        if config.get("cmc", False):
            cmc_adapters = build_masquant_cmc_adapters(
                lisa_model,
                multimodal_inputs,
                act_scales,
                alpha=config.get("alpha", 0.5),
                scale_mode=config.get("scale_mode", "split"),
                w_bit=config.get("w_bit", 4),
                weight_zero_point=config.get("zero_point", False),
                q_group_size=config.get("q_group_size", -1),
                rank=config.get("cmc_rank", 0.05),
                oversample=config.get("cmc_oversample", 32),
                niter=config.get("cmc_niter", 2),
                quant_cmc=config.get("quant_cmc", False),
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                calibration_batch_size=config.get("calibration_batch_size", 1),
            )
        _save_masquant_payload(scale_path, model_path, config, act_scales, cmc_adapters)
    finally:
        if owns_model:
            del lisa_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return str(scale_path)


def load_masquant_backbone_into_lisa(
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
        raise ValueError("quant_kwargs for MASQuant must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("MASQuant quant_kwargs must include '_config_path'.")

    config = load_masquant_config(config_path)
    if torch_dtype is None:
        runtime_device = _get_runtime_device(device)
        torch_dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32

    scale_path = ensure_masquant_scales(
        model_path,
        quant_kwargs,
        force=force,
        runtime_device=device,
        runtime_dtype=torch_dtype,
        lisa_model=lisa_model,
    )
    payload = _load_masquant_payload(scale_path)
    act_scales = payload["act_scales"]
    cmc_adapters = payload["cmc_adapters"]
    if config.get("cmc", False) and not cmc_adapters:
        raise ValueError(f"MASQuant CMC is enabled, but no CMC adapters were found in {scale_path}.")

    pseudo_quantize_masquant_backbone(
        lisa_model.get_model(),
        act_scales,
        alpha=config.get("alpha", 0.5),
        scale_mode=config.get("scale_mode", "split"),
        wa_quant=config.get("wa_quant", False),
        w_bit=config.get("w_bit", 4),
        a_bit=config.get("a_bit", 8),
        weight_zero_point=config.get("zero_point", False),
        q_group_size=config.get("q_group_size", -1),
        cmc_adapters=cmc_adapters if config.get("cmc", False) else None,
        act_quant_mode=config.get("activation_quant_mode", "prefill"),
    )
    lisa_model.quantization_method = "masquant"
    lisa_model.config.quantization_config = {
        "quant_method": "masquant",
        "w_bit": config.get("w_bit", 4),
        "a_bit": config.get("a_bit", 8),
        "wa_quant": bool(config.get("wa_quant", False)),
        "scale_mode": config.get("scale_mode", "split"),
        "alpha": config.get("alpha", 0.5),
        "cmc": bool(config.get("cmc", False)),
        "cmc_rank": config.get("cmc_rank", 0.05),
        "quant_cmc": bool(config.get("quant_cmc", False)),
        "zero_point": bool(config.get("zero_point", False)),
        "q_group_size": config.get("q_group_size", -1),
        "activation_quant_mode": config.get("activation_quant_mode", "prefill"),
    }
    return lisa_model


def main():
    args = parse_args()
    config = load_masquant_config(args.config)
    scale_path = ensure_masquant_scales(
        model_path=config["model_path"],
        quant_kwargs={"_config_path": config["_config_path"]},
        force=args.force,
    )
    print(scale_path)


if __name__ == "__main__":
    main()
