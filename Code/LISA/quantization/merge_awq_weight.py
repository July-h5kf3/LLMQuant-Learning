import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


DEFAULT_AWQ_SUBDIR = "awq_llm"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge an AWQ LLM checkpoint with the original LISA weights."
    )
    parser.add_argument("--base-model-path", required=True, type=str)
    parser.add_argument("--awq-model-path", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--awq-subdir", type=str, default=DEFAULT_AWQ_SUBDIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write the AWQ merge bundle into the base model directory.",
    )
    return parser.parse_args()


def link_or_copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


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


def merge_awq_weights(
    base_model_path,
    awq_model_path,
    output_dir=None,
    *,
    awq_subdir=DEFAULT_AWQ_SUBDIR,
    force=False,
    in_place=False,
):
    base_model_path = Path(base_model_path).resolve()
    awq_model_path = Path(awq_model_path).resolve()

    if in_place:
        output_dir = base_model_path
    elif output_dir is None:
        output_dir = base_model_path.parent / f"{base_model_path.name}_awq_merged"
    else:
        output_dir = Path(output_dir).resolve()

    if force and output_dir.exists() and not in_place:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not in_place:
        copy_dir_with_links(base_model_path, output_dir)

    awq_dst_dir = output_dir / awq_subdir
    if force and awq_dst_dir.exists():
        shutil.rmtree(awq_dst_dir)
    copy_dir_with_links(awq_model_path, awq_dst_dir)

    merge_meta = {
        "base_model_path": str(base_model_path),
        "awq_model_path": str(awq_model_path),
        "merged_model_path": str(output_dir),
        "awq_subdir": awq_subdir,
        "in_place": in_place,
    }
    with open(output_dir / "merge_awq_meta.json", "w", encoding="utf-8") as f:
        json.dump(merge_meta, f, indent=2, ensure_ascii=False)

    return str(output_dir)


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


def _replace_named_child(parent_module, child_name, new_child):
    if child_name.isdigit():
        parent_module[int(child_name)] = new_child
    else:
        setattr(parent_module, child_name, new_child)


def _get_named_child(parent_module, child_name):
    if child_name.isdigit():
        return parent_module[int(child_name)]
    return getattr(parent_module, child_name)


def _inject_awq_layer_modules(lisa_layer, awq_layer):
    quantized_linear_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    for module_name in quantized_linear_names:
        lisa_parent = lisa_layer
        awq_parent = awq_layer
        path = module_name.split(".")
        for part in path[:-1]:
            lisa_parent = _get_named_child(lisa_parent, part)
            awq_parent = _get_named_child(awq_parent, part)
        _replace_named_child(lisa_parent, path[-1], _get_named_child(awq_parent, path[-1]))

    lisa_layer.input_layernorm.load_state_dict(awq_layer.input_layernorm.state_dict())
    lisa_layer.post_attention_layernorm.load_state_dict(
        awq_layer.post_attention_layernorm.state_dict()
    )


def _validate_backbone_compatibility(lisa_model, awq_model):
    lisa_backbone = lisa_model.get_model()
    awq_backbone = awq_model.model

    if len(lisa_backbone.layers) != len(awq_backbone.layers):
        raise ValueError(
            "Layer count mismatch between LISA backbone and AWQ backbone: "
            f"{len(lisa_backbone.layers)} != {len(awq_backbone.layers)}"
        )

    if lisa_backbone.embed_tokens.weight.shape != awq_backbone.embed_tokens.weight.shape:
        raise ValueError(
            "Embedding shape mismatch between LISA backbone and AWQ backbone: "
            f"{tuple(lisa_backbone.embed_tokens.weight.shape)} != "
            f"{tuple(awq_backbone.embed_tokens.weight.shape)}"
        )

    if lisa_model.lm_head.weight.shape != awq_model.lm_head.weight.shape:
        raise ValueError(
            "LM head shape mismatch between LISA model and AWQ model: "
            f"{tuple(lisa_model.lm_head.weight.shape)} != "
            f"{tuple(awq_model.lm_head.weight.shape)}"
        )


def load_awq_weights_into_lisa(
    lisa_model,
    awq_model_or_merged_dir,
    *,
    awq_subdir=DEFAULT_AWQ_SUBDIR,
    fuse_layers=False,
    device_map=None,
):
    awq_model_or_merged_dir = Path(awq_model_or_merged_dir).resolve()
    direct_awq_config = awq_model_or_merged_dir / "config.json"
    direct_awq_weights = awq_model_or_merged_dir / "model.safetensors"

    if direct_awq_config.exists() and direct_awq_weights.exists():
        awq_model_dir = awq_model_or_merged_dir
    else:
        awq_model_dir = awq_model_or_merged_dir / awq_subdir
        if not awq_model_dir.is_dir():
            raise FileNotFoundError(
                "Missing AWQ checkpoint directory. Expected either a direct AWQ model dir "
                f"or a merged dir containing '{awq_subdir}': {awq_model_or_merged_dir}"
            )

    if device_map is None:
        device_map = {"": "cpu"}

    AutoAWQForCausalLM = _load_autoawq_class()
    awq_wrapper = AutoAWQForCausalLM.from_quantized(
        str(awq_model_dir),
        trust_remote_code=True,
        fuse_layers=fuse_layers,
        device_map=device_map,
        safetensors=True,
    )
    awq_model = awq_wrapper.model
    _validate_backbone_compatibility(lisa_model, awq_model)

    lisa_backbone = lisa_model.get_model()
    awq_backbone = awq_model.model

    lisa_backbone.embed_tokens.load_state_dict(awq_backbone.embed_tokens.state_dict())
    for lisa_layer, awq_layer in zip(lisa_backbone.layers, awq_backbone.layers):
        _inject_awq_layer_modules(lisa_layer, awq_layer)
    lisa_backbone.norm.load_state_dict(awq_backbone.norm.state_dict())
    lisa_model.lm_head.load_state_dict(awq_model.lm_head.state_dict())

    lisa_model.quantization_method = "awq"
    if hasattr(awq_wrapper, "quant_config") and awq_wrapper.quant_config is not None:
        lisa_model.config.quantization_config = awq_wrapper.quant_config.to_transformers_dict()

    return lisa_model


def main():
    args = parse_args()
    merged_dir = merge_awq_weights(
        args.base_model_path,
        args.awq_model_path,
        output_dir=args.output_dir,
        awq_subdir=args.awq_subdir,
        force=args.force,
        in_place=args.in_place,
    )
    print(merged_dir)


if __name__ == "__main__":
    main()
