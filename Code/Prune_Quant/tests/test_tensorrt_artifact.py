import json
from pathlib import Path

import pytest

from prune_quant_baseline.quant.tensorrt import (
    ARTIFACT_FORMAT,
    format_tensorrt_builder_command,
    load_masquant_tensorrt_artifact,
    write_masquant_tensorrt_artifact,
)
from prune_quant_baseline.scripts.build_masquant_tensorrt import build_arg_parser, build_commands


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
    low_rank = tmp_path / "low_rank_adapters.pt"
    low_rank.write_bytes(b"low-rank")
    white = tmp_path / "white_matrix.pt"
    white.write_bytes(b"white")

    artifact = write_masquant_tensorrt_artifact(
        artifact_dir=tmp_path / "artifact",
        model_path="/models/Qwen2.5-VL-7B-Instruct",
        model_type="qwen2_5_vl",
        engine_dir=engine_dir,
        masquant_resume=resume,
        masquant_act_scales=act_scales,
        cmc_low_rank_adapters=low_rank,
        cmc_white_matrix=white,
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
    assert (artifact.root / manifest["masquant"]["cmc_low_rank_adapters"]).exists()
    assert (artifact.root / manifest["masquant"]["cmc_white_matrix"]).exists()
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


def test_build_masquant_tensorrt_accepts_custom_commands(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--model",
            "/models/Qwen2-VL-7B-Instruct",
            "--masquant-root",
            "/ext/masquant",
            "--masquant-resume",
            "/work/mas_parameters.pth",
            "--cmc-low-rank",
            "/work/low_rank.pt",
            "--output",
            str(tmp_path / "engine"),
            "--convert-command",
            "python convert.py --model {hf_export_dir} --out {checkpoint_dir}",
            "--llm-build-command",
            "python build_llm.py --checkpoint {checkpoint_dir} --out {llm_engine_dir}",
            "--vision-build-command",
            "python build_vision.py --model {hf_export_dir} --out {vision_engine_dir}",
        ]
    )
    values = {
        "hf_export_dir": tmp_path / "work" / "masquant_export" / "hf_model",
        "torch_export_dir": tmp_path / "work" / "masquant_export",
        "checkpoint_dir": tmp_path / "work" / "trtllm_checkpoint",
        "engine_dir": tmp_path / "engine",
        "llm_engine_dir": tmp_path / "engine" / "llm",
        "vision_engine_dir": tmp_path / "engine" / "vision",
        "dtype": "float16",
        "model": args.model,
        "masquant_resume": args.masquant_resume,
        "act_scales": "",
        "cmc_low_rank": args.cmc_low_rank,
        "cmc_white_matrix": "",
    }

    commands = build_commands(args, values)

    assert commands[0][:2] == ["python", "convert.py"]
    assert str(values["checkpoint_dir"]) in commands[0]
    assert str(values["llm_engine_dir"]) in commands[1]
    assert str(values["vision_engine_dir"]) in commands[2]
