import argparse
import sys
from pathlib import Path
from types import MethodType

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantization.quantization_utils import (
    build_calibration_data,
    build_quant_artifact_dirs,
    checkpoint_exists,
    cleanup_export_dir,
    export_lisa_lm_backbone,
    finalize_quantized_output,
    load_method_quant_config,
    patch_transformers_compat,
    prepare_multimodal_gptq_dataset,
    remove_dir,
    reset_dir,
    split_multimodal_calibration_inputs,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the LISA LLM backbone with GPTQ.")
    parser.add_argument(
        "--config",
        default="configs/quant/gptq.yaml",
        type=str,
        help="Path to the GPTQ YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the GPTQ checkpoint even if it already exists.",
    )
    return parser.parse_args()


def load_gptq_config(config_path):
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=(
            "model_path",
            "gptq_model_path",
            "dataset_dir",
            "vision_tower",
            "vision_pretrained",
        ),
        bool_defaults={"keep_lm_export": False},
    )


def save_gptq_metadata(output_dir, model_path, config, calibration_records):
    gptq_model_path = Path(config["gptq_model_path"])
    write_json(
        Path(output_dir) / "gptq_job_meta.json",
        {
            "source_model_path": str(model_path),
            "gptq_model_path": str(output_dir),
            "gptq_source_lm_path": str(
                gptq_model_path.parent / f"{gptq_model_path.name}_lm_backbone"
            ),
            "bits": config["bits"],
            "group_size": config["group_size"],
            "damp_percent": config["damp_percent"],
            "desc_act": config["desc_act"],
            "act_group_aware": config["act_group_aware"],
            "sym": config["sym"],
            "true_sequential": config["true_sequential"],
            "backend": config["backend"],
            "num_calibration_records": len(calibration_records),
            "multimodal_calibration": True,
            "config_path": config["_config_path"],
        },
    )


def run_gptq_quantization(source_lm_dir, output_dir, tokenizer, multimodal_inputs, config):
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from gptqmodel.quantization import FORMAT

    calibration_dataset = split_multimodal_calibration_inputs(multimodal_inputs)

    quant_model = GPTQModel.from_pretrained(
        str(source_lm_dir),
        quantize_config=QuantizeConfig(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            damp_percent=config.get("damp_percent", 0.1),
            desc_act=config.get("desc_act", False),
            act_group_aware=config.get("act_group_aware", True),
            sym=config.get("sym", True),
            true_sequential=config.get("true_sequential", True),
            format=FORMAT(config.get("format", "gptq")),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        ),
        trust_remote_code=True,
    )
    quant_model.prepare_dataset = MethodType(prepare_multimodal_gptq_dataset, quant_model)
    quant_model.quantize(
        calibration_dataset,
        tokenizer=tokenizer,
        batch_size=config.get("batch_size", 1),
        backend=BACKEND(config.get("backend", "auto")),
    )
    quant_model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def ensure_gptq_checkpoint(model_path, quant_kwargs, force=False):
    patch_transformers_compat()

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs for GPTQ must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("GPTQ quant_kwargs must include '_config_path'.")

    config = load_gptq_config(config_path)
    output_dir = Path(config["gptq_model_path"])
    source_lm_dir, temp_output_dir = build_quant_artifact_dirs(output_dir)

    if checkpoint_exists(output_dir, "config.json", "tokenizer_config.json") and not force:
        cleanup_export_dir(source_lm_dir, keep_export=config["keep_lm_export"])
        remove_dir(temp_output_dir)
        print(f"GPTQ checkpoint already exists, reusing: {output_dir}")
        return str(output_dir)

    print("Building multimodal calibration dataset from ReasonSeg...")
    tokenizer, calibration_records, multimodal_inputs = build_calibration_data(
        str(model_path), config
    )
    if not calibration_records:
        raise ValueError("Calibration dataset is empty.")

    remove_dir(temp_output_dir)
    reset_dir(temp_output_dir)
    try:
        print(f"Exporting LISA LLM backbone to: {source_lm_dir}")
        source_lm_dir = export_lisa_lm_backbone(model_path, source_lm_dir, force=force)

        print(f"Running GPTQ quantization with backend={config['backend']}...")
        run_gptq_quantization(
            source_lm_dir,
            temp_output_dir,
            tokenizer,
            multimodal_inputs,
            config,
        )

        save_gptq_metadata(temp_output_dir, model_path, config, calibration_records)
        finalize_quantized_output(temp_output_dir, output_dir)
    finally:
        cleanup_export_dir(source_lm_dir, keep_export=config["keep_lm_export"])
        remove_dir(temp_output_dir)

    print(f"GPTQ checkpoint saved to: {output_dir}")
    return str(output_dir)


def main():
    args = parse_args()
    config = load_gptq_config(args.config)
    ensure_gptq_checkpoint(
        model_path=config["model_path"],
        quant_kwargs={"_config_path": config["_config_path"]},
        force=args.force,
    )


if __name__ == "__main__":
    main()
