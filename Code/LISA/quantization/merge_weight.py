import argparse
import importlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantization.quantization_utils import (
    checkpoint_exists,
    link_or_copy_file,
    load_quantized_backbone_into_lisa,
    load_method_quant_config,
    patch_transformers_compat,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge an external quantized LLM backbone with the original LISA weights."
    )
    parser.add_argument("--config", default="configs/quant/awq.yaml", type=str)
    parser.add_argument("--quant-method", default="awq", choices=("awq", "gptq"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write the merge bundle into the base model directory.",
    )
    return parser.parse_args()



def copy_dir_with_links(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        dst_path = dst_dir / path.name
        if path.is_dir():
            copy_dir_with_links(path, dst_path)
        elif path.is_file():
            link_or_copy_file(path, dst_path)
        elif path.is_symlink():
            target = path.resolve()
            if target.is_dir():
                copy_dir_with_links(target, dst_path)
            elif target.is_file():
                link_or_copy_file(target, dst_path)



def load_merge_config(config_path, *, quant_method="awq"):
    quant_model_key = f"{quant_method}_model_path"
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=("base_model_path", quant_model_key, "merged_model_path"),
    )



def merge_quantized_weights(
    base_model_path,
    quantized_model_path,
    output_dir=None,
    *,
    quant_method,
    quant_subdir,
    force=False,
    in_place=False,
):
    base_model_path = Path(base_model_path).resolve()
    quantized_model_path = Path(quantized_model_path).resolve()

    if in_place:
        output_dir = base_model_path
    elif output_dir is None:
        output_dir = base_model_path.parent / f"{base_model_path.name}_{quant_method}_merged"
    else:
        output_dir = Path(output_dir).resolve()

    if force and output_dir.exists() and not in_place:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not in_place:
        copy_dir_with_links(base_model_path, output_dir)

    quant_dst_dir = output_dir / quant_subdir
    if force and quant_dst_dir.exists():
        shutil.rmtree(quant_dst_dir)
    copy_dir_with_links(quantized_model_path, quant_dst_dir)

    write_json(
        output_dir / "merge_weight_meta.json",
        {
            "base_model_path": str(base_model_path),
            "quantized_model_path": str(quantized_model_path),
            "merged_model_path": str(output_dir),
            "quant_method": quant_method,
            "quant_subdir": quant_subdir,
            "in_place": in_place,
        },
    )

    return str(output_dir)



def _resolve_quantized_model_dir(quant_model_or_merged_dir, *, quant_subdir):
    quant_model_or_merged_dir = Path(quant_model_or_merged_dir).resolve()
    if checkpoint_exists(quant_model_or_merged_dir, "config.json"):
        return quant_model_or_merged_dir

    quant_model_dir = quant_model_or_merged_dir / quant_subdir
    if checkpoint_exists(quant_model_dir, "config.json"):
        return quant_model_dir

    raise FileNotFoundError(
        "Missing quantized checkpoint directory. Expected either a direct quantized model dir "
        f"or a merged dir containing '{quant_subdir}': {quant_model_or_merged_dir}"
    )



def _load_autoawq_class():
    script_dir = str(Path(__file__).resolve().parent)
    removed_entries = []
    while script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_entries.append(script_dir)

    try:
        auto_module = importlib.import_module("awq.models.auto")
        return auto_module.AutoAWQForCausalLM
    finally:
        for entry in reversed(removed_entries):
            sys.path.insert(0, entry)



def load_awq_weights_into_lisa(
    lisa_model,
    awq_model_or_merged_dir,
    *,
    awq_subdir="awq_llm",
    fuse_layers=False,
    device_map=None,
):
    awq_model_dir = _resolve_quantized_model_dir(
        awq_model_or_merged_dir,
        quant_subdir=awq_subdir,
    )

    AutoAWQForCausalLM = _load_autoawq_class()
    awq_wrapper = AutoAWQForCausalLM.from_quantized(
        str(awq_model_dir),
        trust_remote_code=True,
        fuse_layers=fuse_layers,
        device_map=device_map or {"": "cpu"},
        safetensors=True,
    )
    awq_model = awq_wrapper.model
    awq_quant_config = None
    if hasattr(awq_wrapper, "quant_config") and awq_wrapper.quant_config is not None:
        awq_quant_config = awq_wrapper.quant_config.to_transformers_dict()

    return load_quantized_backbone_into_lisa(
        lisa_model,
        awq_model,
        quantization_method="awq",
        quantization_config=awq_quant_config,
    )



def load_gptq_weights_into_lisa(
    lisa_model,
    gptq_model_or_merged_dir,
    *,
    gptq_subdir="gptq_llm",
    backend="auto",
    device="cpu",
):
    patch_transformers_compat()

    from gptqmodel import BACKEND, GPTQModel

    gptq_model_dir = _resolve_quantized_model_dir(
        gptq_model_or_merged_dir,
        quant_subdir=gptq_subdir,
    )

    gptq_wrapper = GPTQModel.from_quantized(
        str(gptq_model_dir),
        device=device,
        backend=BACKEND(backend),
        trust_remote_code=True,
    )
    gptq_model = gptq_wrapper.model
    quantization_config = getattr(
        getattr(gptq_model, "config", None),
        "quantization_config",
        None,
    )

    return load_quantized_backbone_into_lisa(
        lisa_model,
        gptq_model,
        quantization_method="gptq",
        quantization_config=quantization_config,
    )



def main():
    args = parse_args()
    config = load_merge_config(args.config, quant_method=args.quant_method)
    quant_model_key = f"{args.quant_method}_model_path"
    default_subdir = f"{args.quant_method}_llm"
    merge_dir = merge_quantized_weights(
        config["base_model_path"],
        config[quant_model_key],
        output_dir=config.get("merged_model_path"),
        quant_method=args.quant_method,
        quant_subdir=config.get(f"{args.quant_method}_subdir", default_subdir),
        force=args.force,
        in_place=args.in_place,
    )
    print(merge_dir)


if __name__ == "__main__":
    main()
