from functools import partial

import torch
from torch import nn


@torch.no_grad()
def pseudo_quantize_tensor(
    tensor,
    *,
    n_bits=8,
    zero_point=True,
    q_group_size=-1,
    per_tensor=False,
):
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    if q_group_size > 0:
        if original_shape[-1] % q_group_size != 0:
            raise ValueError(
                "q_group_size must divide the last tensor dimension for pseudo quantization."
            )
        work_tensor = tensor.reshape(-1, q_group_size)
    elif per_tensor:
        work_tensor = tensor.reshape(1, -1)
    else:
        work_tensor = tensor.reshape(-1, original_shape[-1])

    work_tensor = work_tensor.to(torch.float32)

    if zero_point:
        max_val = work_tensor.amax(dim=1, keepdim=True)
        min_val = work_tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = work_tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    quantized_tensor = (
        torch.clamp(torch.round(work_tensor / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    quantized_tensor = quantized_tensor.reshape(original_shape).to(original_dtype)

    if torch.isnan(quantized_tensor).any():
        raise ValueError("Pseudo quantization produced NaN values.")

    return quantized_tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(weight, *, n_bits=8):
    return pseudo_quantize_tensor(
        weight,
        n_bits=n_bits,
        zero_point=False,
        q_group_size=-1,
        per_tensor=False,
    )


@torch.no_grad()
def quantize_weight_per_tensor_absmax(weight, *, n_bits=8):
    return pseudo_quantize_tensor(
        weight,
        n_bits=n_bits,
        zero_point=False,
        q_group_size=-1,
        per_tensor=True,
    )


@torch.no_grad()
def quantize_activation_per_token_absmax(activation, *, n_bits=8):
    activation_shape = activation.shape
    quantized = pseudo_quantize_tensor(
        activation.reshape(-1, activation_shape[-1]),
        n_bits=n_bits,
        zero_point=False,
        q_group_size=-1,
        per_tensor=False,
    )
    return quantized.reshape(activation_shape)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(activation, *, n_bits=8):
    activation_shape = activation.shape
    quantized = pseudo_quantize_tensor(
        activation.reshape(-1, activation_shape[-1]),
        n_bits=n_bits,
        zero_point=False,
        q_group_size=-1,
        per_tensor=True,
    )
    return quantized.reshape(activation_shape)


class PseudoQuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=True,
        act_quant="per_token",
        a_bit=8,
        w_bit=8,
        quantize_output=False,
        act_quant_mode="always",
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_bit = a_bit
        self.w_bit = w_bit
        if act_quant_mode not in {"always", "prefill"}:
            raise ValueError(
                f"Unsupported activation quantization policy: {act_quant_mode}"
            )
        self.act_quant_mode = act_quant_mode
        self.activation_stage = "prefill"

        self.register_buffer(
            "weight",
            torch.zeros(
                (out_features, in_features),
                dtype=dtype,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=dtype, device=device),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=a_bit)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=a_bit)
        else:
            raise ValueError(f"Unsupported activation quantization mode: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "none"
            self.output_quant = lambda x: x

        self.weight_quant_name = "unknown"

    @torch.no_grad()
    def forward(self, x):
        quantize_activation = self.should_quantize_activation()
        quantized_x = self.act_quant(x) if quantize_activation else x
        weight = self.weight
        bias = self.bias
        if weight.dtype != quantized_x.dtype:
            weight = weight.to(dtype=quantized_x.dtype)
        if bias is not None and bias.dtype != quantized_x.dtype:
            bias = bias.to(dtype=quantized_x.dtype)
        output = torch.nn.functional.linear(quantized_x, weight, bias)
        if quantize_activation:
            return self.output_quant(output)
        return output

    def set_activation_stage(self, stage):
        if stage not in {"prefill", "decode"}:
            raise ValueError(f"Unsupported activation stage: {stage}")
        self.activation_stage = stage

    def should_quantize_activation(self):
        if self.a_bit >= 16:
            return False
        if self.act_quant_mode == "always":
            return True
        return self.activation_stage == "prefill"

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        module,
        *,
        weight_quant="per_channel",
        act_quant="per_token",
        w_bit=4,
        a_bit=8,
        weight_group=128,
        quantize_output=False,
        act_quant_mode="always",
    ):
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"PseudoQuantLinear can only be created from nn.Linear, got {type(module)}"
            )

        new_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            act_quant=act_quant,
            a_bit=a_bit,
            w_bit=w_bit,
            quantize_output=quantize_output,
            act_quant_mode=act_quant_mode,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )

        if weight_quant == "per_channel":
            quantized_weight = quantize_weight_per_channel_absmax(
                module.weight.data,
                n_bits=w_bit,
            )
        elif weight_quant == "per_tensor":
            quantized_weight = quantize_weight_per_tensor_absmax(
                module.weight.data,
                n_bits=w_bit,
            )
        elif weight_quant == "per_group":
            quantized_weight = pseudo_quantize_tensor(
                module.weight.data,
                n_bits=w_bit,
                zero_point=False,
                q_group_size=weight_group,
                per_tensor=False,
            )
        else:
            raise ValueError(f"Unsupported weight quantization mode: {weight_quant}")

        new_module.weight.copy_(quantized_weight)
        new_module.weight_quant_name = weight_quant

        if module.bias is not None:
            new_module.bias.copy_(module.bias.data)

        return new_module

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, w_bit={self.w_bit}, a_bit={self.a_bit}, "
            f"weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, "
            f"act_quant_mode={self.act_quant_mode}, activation_stage={self.activation_stage}"
        )
