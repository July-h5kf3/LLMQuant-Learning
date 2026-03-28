import math
import warnings
from functools import partial
from typing import Optional, Union

import torch
from torch import nn

from .norm import NORM_CLASS_REGISTRY


def fused_init_helper_(module: nn.Module, init_fn_):
    fused = getattr(module, "_fused", None)
    assert fused is not None
    dim, splits = fused
    splits = (0, *splits, module.weight.size(dim))
    for start, end in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim
        slice_indices[dim] = slice(start, end)
        init_fn_(module.weight[slice_indices])


def generic_param_init_fn_(
    module: nn.Module,
    init_fn_,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim=None,
    verbose: int = 0,
    **kwargs,
):
    del kwargs, emb_init_uniform_lim

    assert (
        init_div_is_residual is False
        or init_div_is_residual is True
        or isinstance(init_div_is_residual, (float, int))
        or (
            isinstance(init_div_is_residual, str)
            and init_div_is_residual.isnumeric()
        )
    )

    if init_div_is_residual is False:
        residual_div = 1.0
    elif init_div_is_residual is True:
        residual_div = math.sqrt(2 * n_layers)
    elif isinstance(init_div_is_residual, str) and init_div_is_residual.isnumeric():
        residual_div = float(init_div_is_residual)
    else:
        residual_div = init_div_is_residual

    if isinstance(module, nn.Linear):
        if hasattr(module, "_fused"):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if init_div_is_residual is not False and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(residual_div)
    elif isinstance(module, nn.Embedding):
        if emb_init_std is not None:
            if verbose > 1:
                warnings.warn(
                    f"Embedding layer initialized using normal distribution with std={emb_init_std!r}."
                )
            nn.init.normal_(module.weight, mean=0.0, std=emb_init_std)
        else:
            init_fn_(module.weight)
    elif isinstance(module, tuple(set(NORM_CLASS_REGISTRY.values()))):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert d_model is not None
            splits = (0, d_model, 2 * d_model, 3 * d_model)
            for start, end in zip(splits[:-1], splits[1:]):
                init_fn_(module.in_proj_weight[start:end])
        else:
            assert module.q_proj_weight is not None
            assert module.k_proj_weight is not None
            assert module.v_proj_weight is not None
            init_fn_(module.q_proj_weight)
            init_fn_(module.k_proj_weight)
            init_fn_(module.v_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            nn.init.zeros_(module.bias_v)
        init_fn_(module.out_proj.weight)
        if init_div_is_residual is not False and getattr(
            module.out_proj, "_is_residual", False
        ):
            with torch.no_grad():
                module.out_proj.weight.div_(residual_div)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)
    else:
        assert len(list(module.parameters(recurse=False))) == 0


def kaiming_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim=None,
    init_gain: float = 0.0,
    fan_mode: str = "fan_in",
    init_nonlinearity: str = "relu",
    verbose: int = 0,
    **kwargs,
):
    del kwargs, emb_init_uniform_lim
    if verbose > 1:
        warnings.warn(
            "Using nn.init.kaiming_normal_ with "
            + f"a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}"
        )
    init_fn_ = partial(
        nn.init.kaiming_normal_,
        a=init_gain,
        mode=fan_mode,
        nonlinearity=init_nonlinearity,
    )
    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        verbose=verbose,
    )


MODEL_INIT_REGISTRY = {
    "kaiming_normal_": kaiming_normal_param_init_fn_,
}
