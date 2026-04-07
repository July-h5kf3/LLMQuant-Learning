from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, QuantoConfig

from quantization.quantization_utils import (
    build_lisa_tokenizer,
    build_quant_artifact_dirs,
    checkpoint_exists,
    cleanup_export_dir,
    export_lisa_lm_backbone,
    finalize_quantized_output,
    load_quantized_backbone_into_lisa,
    load_method_quant_config,
    patch_transformers_compat,
    remove_dir,
    reset_dir,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_BACKBONE_METHODS = {"hqq", "quanto"}


def load_hf_quant_config(config_path, quant_method):
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=("model_path", f"{quant_method}_model_path"),
        bool_defaults={"keep_lm_export": False},
    )



def _save_quant_job_meta(output_dir, model_path, quant_method, config):
    write_json(
        Path(output_dir) / f"{quant_method}_job_meta.json",
        {
            "source_model_path": str(model_path),
            "quantized_model_path": str(output_dir),
            "quant_method": quant_method,
            "config_path": config["_config_path"],
        },
    )



def _build_quanto_config(config):
    return QuantoConfig(
        weights=config.get("weights", "int4"),
        activations=config.get("activations"),
        modules_to_not_convert=config.get("modules_to_not_convert"),
    )



def _build_hqq_quant_config(config):
    from hqq.core.quantize import BaseQuantizeConfig

    return BaseQuantizeConfig(
        nbits=config.get("bits", 4),
        group_size=config.get("group_size", 64),
        axis=config.get("axis", 1),
        view_as_float=config.get("view_as_float", False),
        quant_zero=config.get("quant_zero", False),
        quant_scale=config.get("quant_scale", False),
        offload_meta=config.get("offload_meta", False),
    )



def _build_hqq_quantization_meta(config):
    return {
        "quant_method": "hqq",
        "bits": config.get("bits", 4),
        "group_size": config.get("group_size", 64),
        "axis": config.get("axis", 1),
        "view_as_float": config.get("view_as_float", False),
    }



def _load_hqq_quantized_model(output_dir, runtime_device, runtime_dtype):
    from hqq.models.hf.base import AutoHQQHFModel

    device = runtime_device if runtime_device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return AutoHQQHFModel.from_quantized(
        str(output_dir),
        compute_dtype=runtime_dtype,
        device=device,
    )



def _build_hqq_checkpoint(
    model_path,
    source_lm_dir,
    output_dir,
    temp_output_dir,
    config,
    runtime_device,
    runtime_dtype,
):
    tokenizer = build_lisa_tokenizer(model_path, config["model_max_length"])
    remove_dir(temp_output_dir)
    reset_dir(temp_output_dir)

    try:
        from hqq.models.hf.base import AutoHQQHFModel

        quant_model = AutoModelForCausalLM.from_pretrained(
            str(source_lm_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=runtime_dtype,
        )
        AutoHQQHFModel.quantize_model(
            quant_model,
            _build_hqq_quant_config(config),
            compute_dtype=runtime_dtype,
            device=runtime_device,
        )
        AutoHQQHFModel.save_quantized(quant_model, str(temp_output_dir))
        tokenizer.save_pretrained(str(temp_output_dir))
        _save_quant_job_meta(temp_output_dir, model_path, "hqq", config)
        finalize_quantized_output(temp_output_dir, output_dir)
    finally:
        remove_dir(temp_output_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _load_hqq_quantized_model(output_dir, runtime_device, runtime_dtype)



def _build_quanto_runtime_model(source_lm_dir, config, runtime_dtype):
    return AutoModelForCausalLM.from_pretrained(
        str(source_lm_dir),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=runtime_dtype,
        quantization_config=_build_quanto_config(config),
    )



def prepare_hf_quantized_backbone(
    model_path,
    quant_method,
    quant_kwargs,
    force=False,
    runtime_device=None,
    runtime_dtype=torch.float16,
):
    if quant_method not in HF_BACKBONE_METHODS:
        raise ValueError(
            f"Unsupported HF backbone quantization method: {quant_method}. "
            f"Supported values: {sorted(HF_BACKBONE_METHODS)}"
        )
    if not isinstance(quant_kwargs, dict):
        raise ValueError(f"quant_kwargs for {quant_method} must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError(f"{quant_method} quant_kwargs must include '_config_path'.")

    patch_transformers_compat()
    config = load_hf_quant_config(config_path, quant_method)
    output_key = f"{quant_method}_model_path"
    output_dir = Path(
        config.get(output_key, Path(model_path).with_name(f"{Path(model_path).name}_{quant_method}"))
    )
    source_lm_dir, temp_output_dir = build_quant_artifact_dirs(output_dir)

    if runtime_device is None:
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = str(runtime_device)

    if quant_method == "hqq" and checkpoint_exists(output_dir, "config.json", "qmodel.pt") and not force:
        cleanup_export_dir(source_lm_dir, keep_export=config["keep_lm_export"])
        remove_dir(temp_output_dir)
        return _load_hqq_quantized_model(output_dir, runtime_device, runtime_dtype)

    remove_dir(temp_output_dir)
    export_lisa_lm_backbone(model_path, source_lm_dir, force=force)

    try:
        if quant_method == "hqq":
            return _build_hqq_checkpoint(
                model_path,
                source_lm_dir,
                output_dir,
                temp_output_dir,
                config,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
            )
        if quant_method == "quanto":
            return _build_quanto_runtime_model(source_lm_dir, config, runtime_dtype)
    finally:
        cleanup_export_dir(source_lm_dir, keep_export=config["keep_lm_export"])
        if quant_method == "quanto":
            remove_dir(temp_output_dir)

    raise ValueError(f"Unsupported HF backbone quantization method: {quant_method}")



def load_hf_quantized_backbone_into_lisa(
    lisa_model,
    model_path,
    quant_method,
    quant_kwargs,
    force=False,
    device=None,
    torch_dtype=torch.float16,
):
    quantized_backbone = prepare_hf_quantized_backbone(
        model_path,
        quant_method,
        quant_kwargs,
        force=force,
        runtime_device=device,
        runtime_dtype=torch_dtype,
    )
    quantization_config = getattr(
        getattr(quantized_backbone, "config", None),
        "quantization_config",
        None,
    )
    if quantization_config is None and quant_method == "hqq":
        config = load_hf_quant_config(quant_kwargs["_config_path"], quant_method)
        quantization_config = _build_hqq_quantization_meta(config)
    return load_quantized_backbone_into_lisa(
        lisa_model,
        quantized_backbone,
        quantization_method=quant_method,
        quantization_config=quantization_config,
    )
