import argparse
import importlib
import sys
from pathlib import Path

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
    remove_dir,
    reset_dir,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the LISA LLM backbone with AWQ.")
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


def load_awq_config(config_path):
    return load_method_quant_config(
        config_path,
        base_dir=REPO_ROOT,
        path_keys=(
            "model_path",
            "awq_model_path",
            "dataset_dir",
            "vision_tower",
            "vision_pretrained",
        ),
        bool_defaults={"keep_lm_export": False},
    )


def run_autoawq_quantization(model_path, output_dir, tokenizer, multimodal_inputs, config):
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
        quantizer_module = importlib.import_module("awq.quantize.quantizer")
        AutoAWQForCausalLM = auto_module.AutoAWQForCausalLM
        AwqQuantizer = quantizer_module.AwqQuantizer
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

    class MultimodalAwqQuantizer(AwqQuantizer):
        def __init__(
            self,
            *args,
            calib_inputs_embeds=None,
            calib_attention_mask=None,
            calib_position_ids=None,
            **kwargs,
        ):
            self.calib_inputs_embeds = calib_inputs_embeds
            self.calib_attention_mask = calib_attention_mask
            self.calib_position_ids = calib_position_ids
            super().__init__(*args, **kwargs)

        def init_quant(self, n_samples=128, max_seq_len=512):
            if self.calib_inputs_embeds is None:
                return super().init_quant(n_samples=n_samples, max_seq_len=max_seq_len)

            modules = self.awq_model.get_model_layers(self.model)
            layer_kwargs = {"use_cache": False}

            if self.calib_attention_mask is not None:
                layer_kwargs["attention_mask"] = self.calib_attention_mask
            if self.calib_position_ids is not None:
                layer_kwargs["position_ids"] = self.calib_position_ids

            return modules, layer_kwargs, self.calib_inputs_embeds

    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=False,
    )
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=["multimodal-calibration"],
        duo_scaling=config["duo_scaling"],
        max_calib_samples=multimodal_inputs["num_samples"],
        max_calib_seq_len=multimodal_inputs["max_seq_len"],
        quantizer_cls=MultimodalAwqQuantizer,
        calib_inputs_embeds=multimodal_inputs["inputs_embeds"],
        calib_attention_mask=multimodal_inputs["attention_mask"],
        calib_position_ids=multimodal_inputs["position_ids"],
    )
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def save_awq_metadata(output_dir, model_path, config, calibration_records):
    awq_model_path = Path(config["awq_model_path"])
    write_json(
        Path(output_dir) / "awq_job_meta.json",
        {
            "source_model_path": str(model_path),
            "awq_model_path": str(output_dir),
            "awq_source_lm_path": str(awq_model_path.parent / f"{awq_model_path.name}_lm_backbone"),
            "backend": config["backend"],
            "bits": config["bits"],
            "group_size": config["group_size"],
            "zero_point": config["zero_point"],
            "version": config["version"],
            "num_calibration_records": len(calibration_records),
            "multimodal_calibration": True,
            "config_path": config["_config_path"],
        },
    )


def ensure_awq_checkpoint(model_path, quant_kwargs, force=False):
    patch_transformers_compat()

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs for AWQ must be a mapping.")

    config_path = quant_kwargs.get("_config_path")
    if not config_path:
        raise ValueError("AWQ quant_kwargs must include '_config_path'.")

    config = load_awq_config(config_path)
    output_dir = Path(config["awq_model_path"])
    source_lm_dir, temp_output_dir = build_quant_artifact_dirs(output_dir)

    if checkpoint_exists(output_dir, "config.json", "tokenizer_config.json") and not force:
        cleanup_export_dir(source_lm_dir, keep_export=config["keep_lm_export"])
        remove_dir(temp_output_dir)
        print(f"AWQ checkpoint already exists, reusing: {output_dir}")
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

        if config["backend"] != "autoawq":
            raise ValueError(f"Unsupported AWQ backend: {config['backend']}")

        print(f"Running AWQ quantization with backend={config['backend']}...")
        run_autoawq_quantization(
            str(source_lm_dir),
            temp_output_dir,
            tokenizer,
            multimodal_inputs,
            config,
        )

        save_awq_metadata(temp_output_dir, model_path, config, calibration_records)
        finalize_quantized_output(temp_output_dir, output_dir)
    finally:
        cleanup_export_dir(source_lm_dir, keep_export=config["keep_lm_export"])
        remove_dir(temp_output_dir)

    print(f"AWQ checkpoint saved to: {output_dir}")
    return str(output_dir)


def main():
    args = parse_args()
    config = load_awq_config(args.config)
    ensure_awq_checkpoint(
        model_path=config["model_path"],
        quant_kwargs={"_config_path": config["_config_path"]},
        force=args.force,
    )


if __name__ == "__main__":
    main()
