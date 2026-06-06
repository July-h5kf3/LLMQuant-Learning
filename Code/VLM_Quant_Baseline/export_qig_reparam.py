import argparse
import json
import shutil
import warnings
from pathlib import Path


SUPPORTED_MODELS = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a QIG scale file to a HuggingFace VLM checkpoint and save the "
            "reparameterized full-precision checkpoint. Use this output as the "
            "input to backend real-quant exporters."
        )
    )
    parser.add_argument("--model", default="qwen2_vl", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--model_dir", required=True, help="Original HuggingFace checkpoint path.")
    parser.add_argument("--scale_path", required=True, help="QIG .pt scale file from main_quant.py.")
    parser.add_argument("--output_dir", required=True, help="Output HuggingFace checkpoint directory.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--device_map",
        default="none",
        choices=["none", "cpu", "auto", "sequential"],
        help="Optional transformers device_map for loading the source model.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--w_bit", type=int, default=None, help="Recorded in the manifest only.")
    parser.add_argument("--a_bit", type=int, default=None, help="Recorded in the manifest only.")
    return parser.parse_args()


def ensure_empty_or_forced(path: Path, force: bool):
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"{path} is not empty. Pass --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def torch_dtype_from_string(dtype: str):
    if dtype == "auto":
        return "auto"

    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_qig_results(scale_path: str):
    import torch

    results = torch.load(scale_path, map_location="cpu")
    if not isinstance(results, dict) or "scale" not in results:
        raise ValueError(f"{scale_path} is not a valid QIG result file: missing 'scale'.")
    if not isinstance(results["scale"], list) or not results["scale"]:
        raise ValueError(f"{scale_path} contains no QIG scales.")
    return results


def apply_qig_to_model(model, qig_results, apply_fn=None):
    if not isinstance(qig_results, dict) or "scale" not in qig_results:
        raise ValueError("QIG results must be a dict containing a non-empty 'scale' list.")
    if not isinstance(qig_results["scale"], list) or not qig_results["scale"]:
        raise ValueError("QIG results must contain a non-empty 'scale' list.")

    if apply_fn is None:
        from qmllm.methods.qig.quantize.pre_quant import apply_qig

        apply_fn = apply_qig

    apply_fn(model, qig_results)
    return model


def load_model(model_name: str, model_dir: str, dtype: str, device_map: str, trust_remote_code: bool):
    from transformers import AutoProcessor, AutoTokenizer
    import transformers

    class_name = SUPPORTED_MODELS[model_name]
    model_cls = getattr(transformers, class_name)
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype_from_string(dtype),
        "low_cpu_mem_usage": True,
    }
    if device_map != "none":
        load_kwargs["device_map"] = device_map

    model = model_cls.from_pretrained(model_dir, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    return model, processor, tokenizer


def save_processor_sidecars(processor, tokenizer, output_dir: Path):
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def write_manifest(args, qig_results):
    scale_entries = qig_results.get("scale") or []
    manifest = {
        "source_model_dir": args.model_dir,
        "scale_path": args.scale_path,
        "model": args.model,
        "w_bit": args.w_bit,
        "a_bit": args.a_bit,
        "backend_input_kind": "hf_qig_reparameterized_full_precision",
        "qig_num_scales": len(scale_entries),
        "qig_applied_to": "full_huggingface_model",
        "note": (
            "This checkpoint is not packed low-bit by itself. It is the original "
            "HF model after applying QIG reparameterization scales; feed it into "
            "the TensorRT-LLM, AutoRound, or other backend real-quant exporter."
        ),
    }
    with open(Path(args.output_dir) / "qig_reparam_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def export_qig_reparam(args):
    out = Path(args.output_dir)
    ensure_empty_or_forced(out, args.force)

    qig_results = load_qig_results(args.scale_path)
    model, processor, tokenizer = load_model(
        model_name=args.model,
        model_dir=args.model_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    apply_qig_to_model(model, qig_results)

    model.save_pretrained(out, safe_serialization=True, max_shard_size=args.max_shard_size)
    save_processor_sidecars(processor, tokenizer, out)
    write_manifest(args, qig_results)
    return out


def main():
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        out = export_qig_reparam(args)
    print(f"[OK] Saved QIG reparameterized checkpoint to: {out}")


if __name__ == "__main__":
    main()
