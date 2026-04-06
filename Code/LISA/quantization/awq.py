import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.LISA import LISAForCausalLM
from quantization.calibration_dataset import build_calibration_loader
from utils.tokenizer_compat import add_lisa_seg_token, load_lisa_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the LISA LLM backbone with AWQ.")
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to the original LISA model weights.",
    )
    parser.add_argument(
        "--config",
        default="configs/quant/awq.yaml",
        type=str,
        help="Path to the AWQ YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the AWQ checkpoint even if it already exists.",
    )
    return parser.parse_args()


def load_yaml_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("AWQ config YAML must be a mapping.")

    config["_config_path"] = str(config_path)
    return config


def resolve_path(path_value, *, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def patch_transformers_compat():
    # gptqmodel/awq-related tooling may still import no_init_weights from the old path.
    import transformers
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "no_init_weights"):
        pass
    else:
        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            from transformers.modeling_utils import no_init_weights

        modeling_utils.no_init_weights = no_init_weights

    # AutoAWQ still looks for the old multimodal auto class name.
    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
        transformers, "AutoModelForImageTextToText"
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

    # LISA checkpoints advertise `model_type=llava`, but they are loaded in this repo
    # through a causal LM path rather than a processor-backed vision2seq auto class.
    try:
        from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
    except ImportError:
        return

    if TRANSFORMERS_AUTO_MAPPING_DICT.get("llava") == "AutoModelForVision2Seq":
        TRANSFORMERS_AUTO_MAPPING_DICT["llava"] = "AutoModelForCausalLM"


def awq_checkpoint_exists(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return False

    required_files = [
        output_dir / "config.json",
        output_dir / "tokenizer_config.json",
    ]
    if not all(path.exists() for path in required_files):
        return False

    weight_files = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    return any((output_dir / filename).exists() for filename in weight_files)


def ensure_output_dir(output_dir, force=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def cleanup_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def build_lisa_tokenizer(model_path, model_max_length):
    tokenizer = load_lisa_tokenizer(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    add_lisa_seg_token(tokenizer)
    return tokenizer


def exported_checkpoint_exists(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return False

    if not (output_dir / "config.json").exists():
        return False

    return any(
        (output_dir / filename).exists()
        for filename in [
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
    )


def is_lisa_lm_weight(weight_name):
    lm_prefixes = (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
    )
    return weight_name == "lm_head.weight" or weight_name.startswith(lm_prefixes)


def build_exported_lm_config(source_config):
    config = dict(source_config)
    config["architectures"] = ["LlamaForCausalLM"]
    config["model_type"] = "llama"
    config["torch_dtype"] = "float16"
    return config


def estimate_lm_weight_bytes(model_path, lm_weight_map):
    shard_to_names = {}
    for weight_name, shard_name in lm_weight_map.items():
        shard_to_names.setdefault(shard_name, []).append(weight_name)

    total_size = 0
    for shard_name, weight_names in shard_to_names.items():
        shard_state = torch.load(Path(model_path) / shard_name, map_location="cpu")
        for weight_name in weight_names:
            tensor = shard_state[weight_name]
            total_size += tensor.numel() * tensor.element_size()
        del shard_state
    return total_size


def can_materialize_filtered_shards(export_dir, required_bytes, safety_margin_bytes=2 * 1024**3):
    free_bytes = shutil.disk_usage(export_dir).free
    return free_bytes >= required_bytes + safety_margin_bytes


def link_or_copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def export_lisa_lm_backbone(model_path, export_dir, force=False):
    model_path = Path(model_path)
    export_dir = ensure_output_dir(export_dir, force=force)

    if exported_checkpoint_exists(export_dir) and not force:
        return export_dir

    index_path = model_path / "pytorch_model.bin.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        source_index = json.load(f)

    weight_map = source_index["weight_map"]
    lm_weight_map = {
        name: shard_name for name, shard_name in weight_map.items() if is_lisa_lm_weight(name)
    }
    if not lm_weight_map:
        raise ValueError("No LLM backbone weights were found in the LISA checkpoint.")

    shard_filenames = list(dict.fromkeys(lm_weight_map.values()))
    lm_total_size = estimate_lm_weight_bytes(model_path, lm_weight_map)
    storage_mode = "filtered_shards"

    if can_materialize_filtered_shards(export_dir, lm_total_size):
        for shard_name in shard_filenames:
            shard_path = model_path / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")

            shard_state = torch.load(shard_path, map_location="cpu")
            filtered_state = {
                weight_name: shard_state[weight_name]
                for weight_name, weight_shard in lm_weight_map.items()
                if weight_shard == shard_name
            }
            torch.save(
                filtered_state,
                export_dir / shard_name,
                _use_new_zipfile_serialization=False,
            )
            del shard_state
    else:
        storage_mode = "hardlink_or_symlink_full_shards"
        for shard_name in shard_filenames:
            shard_path = model_path / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
            link_or_copy_file(shard_path, export_dir / shard_name)

    with open(model_path / "config.json", "r", encoding="utf-8") as f:
        source_config = json.load(f)
    exported_config = build_exported_lm_config(source_config)
    with open(export_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(exported_config, f, indent=2, ensure_ascii=False)

    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.exists():
        shutil.copy2(generation_config_path, export_dir / "generation_config.json")

    exported_index = {
        "metadata": {"total_size": lm_total_size},
        "weight_map": lm_weight_map,
    }
    with open(export_dir / "pytorch_model.bin.index.json", "w", encoding="utf-8") as f:
        json.dump(exported_index, f, indent=2, ensure_ascii=False)

    export_meta = {
        "source_model_path": str(model_path),
        "exported_model_path": str(export_dir),
        "export_type": "lisa_lm_backbone",
        "num_weights": len(lm_weight_map),
        "num_shards": len(shard_filenames),
        "storage_mode": storage_mode,
        "estimated_lm_bytes": lm_total_size,
    }
    with open(export_dir / "lisa_lm_export_meta.json", "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)

    return export_dir


def collect_calibration_records(loader):
    records = []
    texts = []

    for batch in loader:
        batch_size = len(batch["conversation_list"])
        for idx in range(batch_size):
            input_ids = batch["input_ids"][idx]
            attention_mask = batch["attention_masks"][idx]
            prompt = batch["conversation_list"][idx]

            seq_len = int(attention_mask.sum().item())
            input_ids = input_ids[:seq_len].tolist()
            attention_mask = attention_mask[:seq_len].tolist()

            records.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            texts.append(prompt)

    return records, texts


def validate_source_model(model_path, config):
    # Reuse the repo's custom loading path so the 4.31-trained weights are checked
    # with the same compatibility logic used elsewhere in the project.
    tokenizer = build_lisa_tokenizer(model_path, config["model_max_length"])
    _ = tokenizer

    model = LISAForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        train_mask_decoder=False,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        seg_token_idx=tokenizer.convert_tokens_to_ids("[SEG]"),
        vision_pretrained=config.get("vision_pretrained"),
        vision_tower=config["vision_tower"],
        use_mm_start_end=config["use_mm_start_end"],
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_calibration_data(model_path, config):
    tokenizer = build_lisa_tokenizer(model_path, config["model_max_length"])
    loader = build_calibration_loader(
        base_image_dir=config["dataset_dir"],
        tokenizer=tokenizer,
        vision_tower=config["vision_tower"],
        reason_seg_data=config["calibration_dataset"],
        image_size=config["image_size"],
        max_samples=config["max_calib_samples"],
        questions_per_image=config["questions_per_image"],
        seed=config["seed"],
        conv_type=config["conv_type"],
        use_mm_start_end=config["use_mm_start_end"],
        batch_size=config["calibration_batch_size"],
        num_workers=config["calibration_num_workers"],
        shuffle=False,
    )
    return tokenizer, collect_calibration_records(loader)


def run_autoawq_quantization(model_path, output_dir, tokenizer, calibration_texts, config):
    script_dir = str(Path(__file__).resolve().parent)
    removed_entries = []
    while script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_entries.append(script_dir)

    try:
        existing_awq = sys.modules.get("awq")
        if existing_awq is not None:
            awq_file = getattr(existing_awq, "__file__", "") or ""
            if Path(awq_file).resolve() == Path(__file__).resolve():
                del sys.modules["awq"]

        auto_module = importlib.import_module("awq.models.auto")
        AutoAWQForCausalLM = auto_module.AutoAWQForCausalLM
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "autoawq is required to run AWQ quantization. "
            "Please install it in the current environment first."
        ) from exc
    finally:
        for entry in reversed(removed_entries):
            sys.path.insert(0, entry)

    quant_config = {
        "zero_point": config["zero_point"],
        "q_group_size": config["group_size"],
        "w_bit": config["bits"],
        "version": config["version"],
    }

    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=False,
    )
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_texts,
        duo_scaling=config["duo_scaling"],
        max_calib_samples=config["max_calib_samples"],
        max_calib_seq_len=config["model_max_length"],
    )
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def save_awq_metadata(output_dir, model_path, config, calibration_records):
    metadata = {
        "source_model_path": str(model_path),
        "awq_model_path": str(output_dir),
        "awq_source_lm_path": str(
            Path(config["awq_model_path"]).parent / f"{Path(config['awq_model_path']).name}_lm_backbone"
        ),
        "backend": config["backend"],
        "bits": config["bits"],
        "group_size": config["group_size"],
        "zero_point": config["zero_point"],
        "version": config["version"],
        "num_calibration_records": len(calibration_records),
        "config_path": config["_config_path"],
    }
    with open(Path(output_dir) / "awq_job_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def normalize_config(config):
    config_dir = Path(config["_config_path"]).parent

    normalized = dict(config)
    normalized["awq_model_path"] = str(
        resolve_path(config["awq_model_path"], base_dir=config_dir)
    )
    normalized["dataset_dir"] = str(
        resolve_path(config["dataset_dir"], base_dir=config_dir)
    )
    normalized["vision_tower"] = str(
        resolve_path(config["vision_tower"], base_dir=config_dir)
    )
    normalized["vision_pretrained"] = str(
        resolve_path(config["vision_pretrained"], base_dir=config_dir)
    )
    normalized.setdefault("backend", "autoawq")
    normalized.setdefault("bits", 4)
    normalized.setdefault("group_size", 128)
    normalized.setdefault("zero_point", True)
    normalized.setdefault("version", "GEMM")
    normalized.setdefault("duo_scaling", True)
    normalized.setdefault("calibration_dataset", "ReasonSeg|train")
    normalized.setdefault("image_size", 1024)
    normalized.setdefault("model_max_length", 1024)
    normalized.setdefault("questions_per_image", 1)
    normalized.setdefault("max_calib_samples", 128)
    normalized.setdefault("calibration_batch_size", 1)
    normalized.setdefault("calibration_num_workers", 0)
    normalized.setdefault("seed", 3407)
    normalized.setdefault("conv_type", "llava_v1")
    normalized.setdefault("use_mm_start_end", False)
    normalized.setdefault("keep_lm_export", False)
    return normalized


def ensure_awq_checkpoint(model_path, quant_kwargs, force=False):
    patch_transformers_compat()

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs for AWQ must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("AWQ quant_kwargs must include '_config_path'.")

    config = normalize_config(load_yaml_config(config_path))
    output_dir = Path(config["awq_model_path"])
    source_lm_dir = output_dir.parent / f"{output_dir.name}_lm_backbone"
    temp_output_dir = output_dir.parent / f"{output_dir.name}.tmp"

    if awq_checkpoint_exists(output_dir) and not force:
        cleanup_dir(source_lm_dir)
        cleanup_dir(temp_output_dir)
        print(f"AWQ checkpoint already exists, reusing: {output_dir}")
        return str(output_dir)

    print(f"Validating source model compatibility: {model_path}")
    validate_source_model(str(model_path), config)

    print("Building calibration dataset from ReasonSeg...")
    tokenizer, (calibration_records, calibration_texts) = build_calibration_data(
        str(model_path), config
    )
    if not calibration_records:
        raise ValueError("Calibration dataset is empty.")

    cleanup_dir(temp_output_dir)
    ensure_output_dir(temp_output_dir, force=False)
    try:
        print(f"Exporting LISA LLM backbone to: {source_lm_dir}")
        source_lm_dir = export_lisa_lm_backbone(model_path, source_lm_dir, force=force)

        if config["backend"] != "autoawq":
            raise ValueError(f"Unsupported AWQ backend: {config['backend']}")

        print(f"Running AWQ quantization with backend={config['backend']}...")
        run_autoawq_quantization(
            str(source_lm_dir),
            temp_output_dir,
            tokenizer,
            calibration_texts,
            config,
        )

        save_awq_metadata(temp_output_dir, model_path, config, calibration_records)
        cleanup_dir(output_dir)
        temp_output_dir.replace(output_dir)
    finally:
        if not config["keep_lm_export"] and source_lm_dir.exists():
            shutil.rmtree(source_lm_dir)
        cleanup_dir(temp_output_dir)

    print(f"AWQ checkpoint saved to: {output_dir}")
    return str(output_dir)


def main():
    args = parse_args()
    ensure_awq_checkpoint(
        model_path=Path(args.model_path).resolve(),
        quant_kwargs={"_config_path": args.config},
        force=args.force,
    )


if __name__ == "__main__":
    main()
