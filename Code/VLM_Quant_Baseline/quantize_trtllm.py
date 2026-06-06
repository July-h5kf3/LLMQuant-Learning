import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


SUPPORTED_TRTLLM_FORMATS = {
    "w4a16": "int4_awq",
    "w4a8": "w4a8_awq",
}

SUPPORTED_TRTLLM_MODELS = {
    "qwen2_vl",
    "llava_onevision",
    "vila",
    "llava",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export real-quant checkpoints for TensorRT-LLM."
    )
    parser.add_argument("--model", default="qwen2_vl",
                        choices=["qwen2_vl", "qwen2_5_vl", "llava_onevision", "vila", "llava"])
    parser.add_argument("--model_dir", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--output_dir", required=True, help="Output checkpoint directory.")
    parser.add_argument("--quant_format", required=True, choices=["w3a16", "w4a16", "w4a8"])
    parser.add_argument("--calib_dataset", default="cnn_dailymail",
                        help="Dataset understood by TensorRT-LLM ModelOpt exporter. "
                             "Use ScienceQA/scienceqa for multimodal Qwen2-VL calibration.")
    parser.add_argument("--calib_size", type=int, default=128)
    parser.add_argument("--calib_max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--awq_block_size", type=int, default=128,
                        help="AWQ group size for W4A16/W4A8.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device_map", default="auto", choices=["auto", "sequential", "cpu", "gpu"])
    parser.add_argument("--tokenizer_max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true")

    # AutoRound fallback for W3A16. This creates a real low-bit checkpoint, but
    # current TensorRT-LLM does not expose a W3 TensorRT engine path.
    parser.add_argument("--autoround_format", default="auto_round",
                        help="AutoRound export format for W3A16. Use auto_round by default; "
                             "auto_gptq can be useful for a non-TRT smoke path.")
    parser.add_argument("--autoround_dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--autoround_iters", type=int, default=200)
    parser.add_argument("--autoround_nsamples", type=int, default=128)
    parser.add_argument("--autoround_seqlen", type=int, default=2048)
    parser.add_argument("--autoround_batch_size", type=int, default=1)
    parser.add_argument("--autoround_device_map", default="0")
    parser.add_argument("--autoround_template", default=None)
    parser.add_argument("--autoround_extra_args", default="",
                        help="Extra raw args appended to auto-round-mllm, e.g. '--low_gpu_mem_usage'.")
    return parser.parse_args()


def ensure_empty_or_forced(path: Path, force: bool):
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"{path} is not empty. Pass --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(args, backend: str, extra=None):
    manifest = {
        "model": args.model,
        "source_model_dir": args.model_dir,
        "quant_format": args.quant_format,
        "backend": backend,
        "tensor_parallel_size": args.tp_size,
        "pipeline_parallel_size": args.pp_size,
    }
    if extra:
        manifest.update(extra)
    with open(Path(args.output_dir) / "vlm_quant_real_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _looks_like_plain_qwen_export(config):
    return (
        config.get("architecture") == "Qwen2ForCausalLM"
        or "Qwen2ForCausalLM" in (config.get("architectures") or [])
        or config.get("model_type") == "qwen2"
        or config.get("qwen_type") == "qwen2"
        or config.get("decoder") in {"llama", "qwen2"}
    )


def _is_qwen2vl_text_wrapper_export(config):
    architectures = config.get("architectures") or [config.get("architecture")]
    return (
        "Qwen2VLTextModel" in architectures
        and config.get("model_type") == "qwen2_vl"
        and config.get("qwen_type") == "qwen2_vl"
        and config.get("position_embedding_type") == "mrope"
        and config.get("decoder") == "qwen2_vl"
    )


def _normalize_qwen2vl_text_wrapper_config(config):
    config["architecture"] = "Qwen2VLForConditionalGeneration"
    config["architectures"] = ["Qwen2VLForConditionalGeneration"]
    config.pop("decoder", None)
    return config


def _require_qwen2_vl_trtllm_config(output_dir: Path):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing TensorRT-LLM config: {config_path}")

    config = _load_json(config_path)
    if (output_dir / "config.json.bak_qwen2vl_normalize").exists():
        raise RuntimeError(
            f"{output_dir} contains config.json.bak_qwen2vl_normalize, which means "
            "it was produced by the old unsafe qwen2->qwen2_vl config rewrite. "
            "Delete the directory and re-export with the patched Qwen2-VL path."
        )
    if _looks_like_plain_qwen_export(config):
        raise RuntimeError(
            "TensorRT-LLM exported a plain qwen2/llama checkpoint instead of a "
            "Qwen2-VL/mRoPE checkpoint. Do not normalize this config after the fact; "
            "that builds an engine that runs quickly but generates invalid logits. "
            "Use a TensorRT-LLM/ModelOpt version or monkey patch that exports "
            "Qwen2-VL with qwen_type=qwen2_vl and position_embedding_type=mrope."
        )

    expected = {
        "model_type": "qwen2_vl",
        "qwen_type": "qwen2_vl",
        "position_embedding_type": "mrope",
    }
    mismatches = {
        key: (expected_value, config.get(key))
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    architectures = config.get("architectures") or [config.get("architecture")]
    if "Qwen2VLForConditionalGeneration" not in architectures:
        mismatches["architectures"] = ("Qwen2VLForConditionalGeneration", architectures)
    if mismatches:
        details = ", ".join(
            f"{key}: expected {expected_value!r}, got {actual_value!r}"
            for key, (expected_value, actual_value) in mismatches.items()
        )
        raise RuntimeError(f"Invalid Qwen2-VL TensorRT-LLM export config: {details}")


def normalize_qwen2_vl_trtllm_config(output_dir: Path):
    """Normalize only Qwen2-VL-native text-wrapper exports.

    Earlier experiments rewrote plain qwen2/Qwen2ForCausalLM TensorRT-LLM
    configs into qwen2_vl/mRoPE configs. Those checkpoints can build engines
    but produce garbage generations. This function only accepts exports that
    already carry qwen2_vl/mRoPE semantics and merely replace the temporary
    Qwen2VLTextModel wrapper architecture with TensorRT-LLM's multimodal Qwen2-VL
    architecture name.
    """
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return False

    config = _load_json(config_path)
    if _looks_like_plain_qwen_export(config):
        raise RuntimeError(
            "Refusing to rewrite a plain qwen2/llama TensorRT-LLM export into "
            "Qwen2-VL. Re-export with the patched Qwen2-VL ModelOpt path instead."
        )
    if _is_qwen2vl_text_wrapper_export(config):
        _save_json(config_path, _normalize_qwen2vl_text_wrapper_config(config))
        _require_qwen2_vl_trtllm_config(output_dir)
        return True
    _require_qwen2_vl_trtllm_config(output_dir)
    return False


def _copy_qwen2vl_text_fields(full_config):
    text_config = getattr(full_config, "text_config", None)
    if text_config is None:
        return full_config

    for name in (
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "layer_types",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "rms_norm_eps",
        "sliding_window",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    ):
        if hasattr(text_config, name):
            setattr(full_config, name, getattr(text_config, name))

    rope_parameters = getattr(text_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = getattr(text_config, "rope_scaling", None)
    if rope_parameters is None:
        rope_parameters = {
            "type": "mrope",
            "rope_type": "default",
            "rope_theta": getattr(text_config, "rope_theta", 1000000.0),
            "mrope_section": [16, 24, 24],
        }
    else:
        rope_parameters = copy.deepcopy(rope_parameters)
    rope_parameters.setdefault("type", "mrope")
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters.setdefault("rope_theta", 1000000.0)
    rope_parameters.setdefault("mrope_section", [16, 24, 24])
    full_config.rope_scaling = rope_parameters
    full_config.rope_parameters = rope_parameters
    full_config.rope_theta = rope_parameters["rope_theta"]
    full_config.architectures = ["Qwen2VLForConditionalGeneration"]
    full_config.model_type = "qwen2_vl"
    full_config.qwen_type = "qwen2_vl"
    return full_config


def _patch_qwen2vl_autoconfig_fields(quantize_and_export):
    qglobals = quantize_and_export.__globals__
    auto_config_cls = qglobals.get("AutoConfig")
    if auto_config_cls is None:
        return False

    original_from_pretrained = auto_config_cls.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        config = original_from_pretrained(*args, **kwargs)
        if getattr(config, "model_type", None) == "qwen2_vl":
            return _copy_qwen2vl_text_fields(config)
        return config

    auto_config_cls.from_pretrained = patched_from_pretrained
    return True


def _patch_qwen2_vl_modelopt_export(quantize_and_export, model_dir: str):
    """Keep ModelOpt's exported LLM config Qwen2-VL-aware.

    TensorRT-LLM 1.3's ModelOpt helper loads Qwen2-VL and then strips it down
    to the language model. Some builds then classify that text module as
    llama/qwen2, which silently drops mRoPE. We still export only the language
    model weights, but we attach a full Qwen2-VL config so ModelOpt emits a
    Qwen2VLForConditionalGeneration/qwen2_vl checkpoint.
    """
    qglobals = quantize_and_export.__globals__
    original_get_model = qglobals.get("get_model")
    original_get_model_type = qglobals.get("get_model_type")
    if original_get_model is None or original_get_model_type is None:
        return False
    _patch_qwen2vl_autoconfig_fields(quantize_and_export)

    def patched_get_model(ckpt_path: str, dtype: str = "bfloat16", device: str = "cuda", device_map: str = "auto"):
        if Path(ckpt_path).resolve() != Path(model_dir).resolve():
            return original_get_model(ckpt_path, dtype=dtype, device=device, device_map=device_map)

        from tensorrt_llm._utils import str_dtype_to_torch
        from transformers import AutoConfig, Qwen2VLForConditionalGeneration

        full_config = _copy_qwen2vl_text_fields(
            AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        )
        torch_dtype = str_dtype_to_torch(dtype)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            ckpt_path,
            device_map=device_map if device != "cpu" else "cpu",
            dtype="auto" if dtype == "auto" else torch_dtype,
            trust_remote_code=True,
        )
        lm_head = model.lm_head
        language_model = model.model.language_model
        language_model.lm_head = lm_head
        language_model.config = full_config
        language_model.eval()
        return language_model

    def patched_get_model_type(model):
        if (
            type(model).__name__ == "Qwen2VLTextModel"
            and getattr(getattr(model, "config", None), "model_type", None) == "qwen2_vl"
        ):
            return "qwen2_vl"
        return original_get_model_type(model)

    qglobals["get_model"] = patched_get_model
    qglobals["get_model_type"] = patched_get_model_type
    model_name_map = qglobals.get("MODEL_NAME_PATTERN_MAP")
    if isinstance(model_name_map, dict):
        model_name_map["Qwen2VLTextModel"] = "qwen2_vl"
    return True


def export_trtllm_modelopt(args):
    if args.awq_block_size not in (64, 128):
        raise ValueError("TensorRT-LLM AWQ block size should be 64 or 128.")
    if args.model not in SUPPORTED_TRTLLM_MODELS:
        raise ValueError(
            f"TensorRT-LLM export is not enabled for --model {args.model}. "
            f"Supported models: {', '.join(sorted(SUPPORTED_TRTLLM_MODELS))}. "
            "The upstream ModelOpt exporter currently has an explicit Qwen2-VL path, "
            "but not Qwen2.5-VL/Qwen3-VL."
        )

    os.environ.setdefault("TRT_LLM_NO_LIB_INIT", "1")
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "8.9")

    try:
        from tensorrt_llm.quantization import quantize_and_export
    except ImportError as exc:
        raise ImportError(
            "TensorRT-LLM is required for W4 TensorRT export. Install `tensorrt_llm` "
            "in the target NVIDIA environment."
        ) from exc

    patched_qwen2_vl_export = False
    if args.model == "qwen2_vl":
        patched_qwen2_vl_export = _patch_qwen2_vl_modelopt_export(
            quantize_and_export,
            args.model_dir,
        )
        if not patched_qwen2_vl_export:
            raise RuntimeError("Failed to patch TensorRT-LLM ModelOpt Qwen2-VL export path.")

    qformat = SUPPORTED_TRTLLM_FORMATS[args.quant_format]
    quantize_and_export(
        model_dir=args.model_dir,
        device=args.device,
        calib_dataset=args.calib_dataset,
        dtype=args.dtype,
        qformat=qformat,
        kv_cache_dtype=None,
        calib_size=args.calib_size,
        batch_size=args.batch_size,
        calib_max_seq_length=args.calib_max_seq_length,
        awq_block_size=args.awq_block_size,
        output_dir=args.output_dir,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
        seed=args.seed,
        tokenizer_max_seq_length=args.tokenizer_max_seq_length,
        device_map=args.device_map,
        quantize_lm_head=False,
    )
    normalized_config = False
    if args.model == "qwen2_vl":
        normalized_config = normalize_qwen2_vl_trtllm_config(Path(args.output_dir))
        _require_qwen2_vl_trtllm_config(Path(args.output_dir))
    write_manifest(args, "trtllm-modelopt", {
        "trtllm_qformat": qformat,
        "block_size": args.awq_block_size,
        "awq_block_size": args.awq_block_size,
        "calib_dataset": args.calib_dataset,
        "calib_size": args.calib_size,
        "patched_qwen2_vl_modelopt_export": patched_qwen2_vl_export,
        "normalized_qwen2_vl_config": normalized_config,
    })


def export_w3_autoround(args):
    if args.tp_size != 1 or args.pp_size != 1 or args.cp_size != 1:
        raise ValueError("W3A16 AutoRound export is a single-checkpoint fallback; keep tp/pp/cp size at 1.")
    cmd = [
        "auto-round",
        "--mllm",
        "--model", args.model_dir,
        "--scheme", "W3A16",
        "--bits", "3",
        "--group_size", "128",
        "--format", args.autoround_format,
        "--output_dir", args.output_dir,
        "--dataset", args.autoround_dataset,
        "--iters", str(args.autoround_iters),
        "--nsamples", str(args.autoround_nsamples),
        "--seqlen", str(args.autoround_seqlen),
        "--batch_size", str(args.autoround_batch_size),
        "--device_map", args.autoround_device_map,
    ]
    if args.autoround_template:
        cmd.extend(["--template", args.autoround_template])
    if args.autoround_extra_args:
        cmd.extend(shlex.split(args.autoround_extra_args))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "auto-round was not found. Install AutoRound with `pip install auto-round`."
        ) from exc

    write_manifest(args, "autoround-w3-fallback", {
        "warning": (
            "This is a real W3A16 packed checkpoint, but current TensorRT-LLM "
            "does not advertise W3A16 TensorRT engine support. Use W4A16/W4A8 for TRT-LLM acceleration."
        ),
        "autoround_format": args.autoround_format,
        "autoround_dataset": args.autoround_dataset,
    })


def main():
    args = parse_args()
    out = Path(args.output_dir)
    ensure_empty_or_forced(out, args.force)

    if args.quant_format in SUPPORTED_TRTLLM_FORMATS:
        export_trtllm_modelopt(args)
    elif args.quant_format == "w3a16":
        export_w3_autoround(args)
    else:
        raise ValueError(f"Unsupported quant format: {args.quant_format}")

    print(f"[OK] Saved {args.quant_format.upper()} export to: {args.output_dir}")


if __name__ == "__main__":
    main()
