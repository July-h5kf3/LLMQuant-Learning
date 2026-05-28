from __future__ import annotations

import importlib
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT_FORMAT = "prune_quant_baseline.masquant_tensorrt"
DEFAULT_RUNTIME_CLASS = "prune_quant_baseline.quant.tensorrt.TensorRTLLMRuntime"


@dataclass(frozen=True)
class MASQuantTensorRTArtifact:
    root: Path
    manifest: dict[str, Any]

    def resolve_path(self, key: str) -> Path:
        value = self.manifest[key]
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    @property
    def engine_dir(self) -> Path:
        return self.resolve_path("engine_dir")

    @property
    def processor_dir(self) -> Path | None:
        value = self.manifest.get("processor_dir")
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()


def _relative_or_absolute(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _copy_optional_file(source: str | Path | None, target_dir: Path) -> str | None:
    if source is None:
        return None
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_path.name
    if source_path != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return _relative_or_absolute(target_path, target_dir.parent)


def format_tensorrt_builder_command(command_template: str, values: Mapping[str, Any]) -> list[str]:
    rendered = command_template.format(**{key: str(value) for key, value in values.items()})
    return shlex.split(rendered)


def run_tensorrt_builder_command(command: Sequence[str], *, cwd: str | Path | None = None, dry_run: bool = False) -> None:
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(cwd) if cwd is not None else None, check=True)


def write_masquant_tensorrt_artifact(
    *,
    artifact_dir: str | Path,
    model_path: str,
    model_type: str,
    engine_dir: str | Path,
    masquant_resume: str | Path,
    masquant_act_scales: str | Path | None,
    cmc_low_rank_adapters: str | Path | None = None,
    cmc_white_matrix: str | Path | None = None,
    wbits: int,
    abits: int,
    group_size: int | None,
    inference_mode: str,
    symmetric: bool,
    runtime_class: str = DEFAULT_RUNTIME_CLASS,
    processor_source: str | Path | None = None,
    save_processor: bool = True,
    builder_command: Sequence[str] | None = None,
) -> MASQuantTensorRTArtifact:
    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    engine_path = Path(engine_dir).expanduser().resolve()
    if not engine_path.exists() or not any(engine_path.iterdir()):
        raise FileNotFoundError(f"TensorRT engine directory is empty or missing: {engine_path}")

    masquant_dir = root / "masquant"
    resume_rel = _copy_optional_file(masquant_resume, masquant_dir)
    act_scales_rel = _copy_optional_file(masquant_act_scales, masquant_dir)
    cmc_dir = root / "cmc"
    low_rank_adapters_rel = _copy_optional_file(cmc_low_rank_adapters, cmc_dir)
    white_matrix_rel = _copy_optional_file(cmc_white_matrix, cmc_dir)

    processor_rel = None
    if save_processor:
        from transformers import AutoProcessor

        processor_path = root / "processor"
        processor = AutoProcessor.from_pretrained(
            processor_source or model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor.save_pretrained(processor_path)
        processor_rel = _relative_or_absolute(processor_path, root)

    manifest = {
        "format": ARTIFACT_FORMAT,
        "backend": "tensorrt",
        "model_path": model_path,
        "model_type": model_type,
        "engine_dir": _relative_or_absolute(engine_path, root),
        "processor_dir": processor_rel,
        "runtime_class": runtime_class,
        "masquant": {
            "resume": resume_rel,
            "act_scales": act_scales_rel,
            "wbits": int(wbits),
            "abits": int(abits),
            "group_size": group_size,
            "inference_mode": inference_mode,
            "symmetric": bool(symmetric),
            "cmc_low_rank_adapters": low_rank_adapters_rel,
            "cmc_white_matrix": white_matrix_rel,
        },
    }
    if builder_command is not None:
        manifest["builder_command"] = list(builder_command)

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return MASQuantTensorRTArtifact(root=root, manifest=manifest)


def load_masquant_tensorrt_artifact(artifact_dir: str | Path) -> MASQuantTensorRTArtifact:
    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MASQuant TensorRT artifact manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported artifact format: {manifest.get('format')!r}")
    if manifest.get("backend") != "tensorrt":
        raise ValueError("MASQuant artifact must declare backend='tensorrt'.")
    artifact = MASQuantTensorRTArtifact(root=root, manifest=manifest)
    if not artifact.engine_dir.exists() or not any(artifact.engine_dir.iterdir()):
        raise FileNotFoundError(f"TensorRT engine directory is empty or missing: {artifact.engine_dir}")
    return artifact


def load_tensorrt_runtime(
    artifact: MASQuantTensorRTArtifact,
    *,
    runtime_class: str | None = None,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    class_path = runtime_class or artifact.manifest.get("runtime_class") or DEFAULT_RUNTIME_CLASS
    module_name, _, attr = class_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid TensorRT runtime class path: {class_path!r}")
    module = importlib.import_module(module_name)
    runtime_cls = getattr(module, attr)
    return runtime_cls(artifact=artifact, **dict(runtime_kwargs or {}))


def _tokenizer_from_processor(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _decode_ids(processor: Any, token_ids: Any) -> str:
    tokenizer = _tokenizer_from_processor(processor)
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()
    raise ValueError("Processor/tokenizer does not provide decode or batch_decode.")


def _first_output_sequence(outputs: Any, *, prompt_len: int) -> Any:
    if isinstance(outputs, dict):
        sequence_lengths = outputs.get("sequence_lengths")
        outputs = outputs.get("output_ids", outputs.get("outputs", outputs))
    else:
        sequence_lengths = None
    if isinstance(outputs, dict):
        raise ValueError("TensorRT runtime output did not contain output_ids.")
    if hasattr(outputs, "dim"):
        if outputs.dim() == 3:
            sequence = outputs[0, 0]
        elif outputs.dim() == 2:
            sequence = outputs[0]
        else:
            sequence = outputs
        if sequence_lengths is not None:
            length = sequence_lengths[0, 0] if getattr(sequence_lengths, "dim", lambda: 0)() == 2 else sequence_lengths[0]
            sequence = sequence[: int(length)]
        return sequence[prompt_len:]
    if isinstance(outputs, list):
        first = outputs[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            first = first[0]
        return first[prompt_len:]
    raise ValueError(f"Unsupported TensorRT runtime output type: {type(outputs)!r}")


class TensorRTLLMRuntime:
    """Thin adapter around TensorRT-LLM ModelRunner.

    Qwen2-VL TensorRT engines can differ in how they accept multimodal tensors.
    This adapter passes the processor-produced multimodal tensors through to the
    runner; if a local engine needs a custom calling convention, set
    ``runtime_class`` in the artifact manifest to a project-specific adapter.
    """

    def __init__(self, *, artifact: MASQuantTensorRTArtifact, rank: int = 0, **runner_kwargs: Any) -> None:
        try:
            from tensorrt_llm.runtime import ModelRunner
        except ImportError as exc:
            raise ImportError(
                "TensorRT backend requires TensorRT-LLM. Install tensorrt_llm in the GPU environment."
            ) from exc

        self.artifact = artifact
        self.runner = ModelRunner.from_dir(str(artifact.engine_dir), rank=rank, **runner_kwargs)

    def generate(self, *, inputs: dict[str, Any], processor: Any, max_new_tokens: int) -> str:
        import torch

        input_ids = inputs["input_ids"]
        prompt_len = int(input_ids.shape[-1])
        batch_input_ids = [row.to(dtype=torch.int32, device="cuda") for row in input_ids]
        tokenizer = _tokenizer_from_processor(processor)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", eos_token_id)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "end_id": int(eos_token_id) if eos_token_id is not None else None,
            "pad_id": int(pad_token_id) if pad_token_id is not None else None,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "attention_mask"):
            if key in inputs:
                value = inputs[key]
                if hasattr(value, "to"):
                    value = value.to("cuda")
                generation_kwargs[key] = value
        outputs = self.runner.generate(batch_input_ids, **generation_kwargs)
        torch.cuda.synchronize()
        token_ids = _first_output_sequence(outputs, prompt_len=prompt_len)
        return _decode_ids(processor, token_ids)
