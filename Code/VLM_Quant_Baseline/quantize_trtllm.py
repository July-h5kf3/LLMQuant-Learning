import argparse
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


def normalize_qwen2_vl_trtllm_text_config(output_dir: Path):
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if (
        config.get("architecture") != "Qwen2VLTextModel"
        and config.get("model_type") != "llama"
        and config.get("decoder") != "llama"
    ):
        return False

    backup_path = output_dir / "config.json.bak_qwen2vltext"
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    config["architecture"] = "Qwen2ForCausalLM"
    config["architectures"] = ["Qwen2ForCausalLM"]
    config["model_type"] = "qwen2"
    config["qwen_type"] = "qwen2"
    config.setdefault("seq_length", config.get("max_position_embeddings", 32768))
    config.pop("decoder", None)
    config.pop("text_config", None)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
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
        normalized_config = normalize_qwen2_vl_trtllm_text_config(Path(args.output_dir))
    write_manifest(args, "trtllm-modelopt", {
        "trtllm_qformat": qformat,
        "block_size": args.awq_block_size,
        "awq_block_size": args.awq_block_size,
        "calib_dataset": args.calib_dataset,
        "calib_size": args.calib_size,
        "normalized_qwen2_vl_text_config": normalized_config,
    })


def export_w3_autoround(args):
    if args.tp_size != 1 or args.pp_size != 1 or args.cp_size != 1:
        raise ValueError("W3A16 AutoRound export is a single-checkpoint fallback; keep tp/pp/cp size at 1.")
    cmd = [
        "auto-round-mllm",
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
            "auto-round-mllm was not found. Install AutoRound with `pip install auto-round`."
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
