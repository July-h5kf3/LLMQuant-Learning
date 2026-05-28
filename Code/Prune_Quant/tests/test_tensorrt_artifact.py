import json
from pathlib import Path

import pytest

from prune_quant_baseline.quant.tensorrt import (
    ARTIFACT_FORMAT,
    format_tensorrt_builder_command,
    load_masquant_tensorrt_artifact,
    write_masquant_tensorrt_artifact,
)


def test_format_tensorrt_builder_command_substitutes_paths_with_quotes(tmp_path: Path) -> None:
    command = format_tensorrt_builder_command(
        "python build.py --model {model_path} --out '{engine_dir}' --wbits {wbits}",
        {
            "model_path": "/models/Qwen2.5-VL-7B-Instruct",
            "engine_dir": tmp_path / "engine dir",
            "wbits": 4,
        },
    )

    assert command == [
        "python",
        "build.py",
        "--model",
        "/models/Qwen2.5-VL-7B-Instruct",
        "--out",
        str(tmp_path / "engine dir"),
        "--wbits",
        "4",
    ]


def test_write_and_load_masquant_tensorrt_artifact_manifest(tmp_path: Path) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "rank0.engine").write_bytes(b"engine")
    resume = tmp_path / "mas_parameters.pth"
    resume.write_bytes(b"resume")
    act_scales = tmp_path / "act_scales.pt"
    act_scales.write_bytes(b"scales")

    artifact = write_masquant_tensorrt_artifact(
        artifact_dir=tmp_path / "artifact",
        model_path="/models/Qwen2.5-VL-7B-Instruct",
        model_type="qwen2_5_vl",
        engine_dir=engine_dir,
        masquant_resume=resume,
        masquant_act_scales=act_scales,
        wbits=4,
        abits=8,
        group_size=0,
        inference_mode="split_scales",
        symmetric=True,
        save_processor=False,
        builder_command=["python", "build.py"],
    )

    loaded = load_masquant_tensorrt_artifact(artifact.root)
    manifest = json.loads((artifact.root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["format"] == ARTIFACT_FORMAT
    assert manifest["backend"] == "tensorrt"
    assert loaded.engine_dir == engine_dir.resolve()
    assert (artifact.root / manifest["masquant"]["resume"]).exists()
    assert (artifact.root / manifest["masquant"]["act_scales"]).exists()
    assert manifest["builder_command"] == ["python", "build.py"]


def test_load_masquant_tensorrt_artifact_rejects_non_tensorrt_backend(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": ARTIFACT_FORMAT,
                "backend": "torch",
                "engine_dir": "engine",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backend='tensorrt'"):
        load_masquant_tensorrt_artifact(artifact_dir)
