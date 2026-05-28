from pathlib import Path

import pytest

from prune_quant_baseline.quant.masquant import (
    MASQuantRunConfig,
    build_generate_act_scales_command,
    build_train_command,
    patch_lmclass_attention_implementation,
    patch_lmclass_qwen2_vl_support,
    patch_qwen25_vl_inputs_embeds_masks,
    validate_masquant_root,
)


def _make_masquant_root(tmp_path: Path) -> Path:
    root = tmp_path / "masquant"
    (root / "quantize").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    (root / "generate_act_scale_shift.py").write_text("", encoding="utf-8")
    (root / "quantize" / "masquant.py").write_text("", encoding="utf-8")
    (root / "models" / "LMClass.py").write_text("", encoding="utf-8")
    return root


def test_validate_masquant_root_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MASQuant checkout"):
        validate_masquant_root(tmp_path / "missing")


def test_build_train_command_uses_pruned_cache_and_act_scales_paths(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    config = MASQuantRunConfig(
        masquant_root=root,
        model_path="/models/Qwen2.5-VL-7B-Instruct",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        act_scales_path=tmp_path / "act_scales" / "pruned.pt",
        dataset_type="text-vision-pruned",
        nsamples=32,
        python="python3",
    )

    command = build_train_command(config)

    assert command[:2] == ["python3", str(root / "main.py")]
    assert command[command.index("--cache_dir") + 1] == str((tmp_path / "cache").resolve())
    assert command[command.index("--act-scales") + 1] == str((tmp_path / "act_scales" / "pruned.pt").resolve())
    assert command[command.index("--dataset-type") + 1] == "text-vision-pruned"
    assert "--let" in command
    assert "--loss_multi_modal_mae_alpha" in command


def test_build_generate_act_scales_command_targets_same_cache_namespace(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    config = MASQuantRunConfig(
        masquant_root=root,
        model_path="/models/Qwen2.5-VL-7B-Instruct",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        dataset_type="text-vision",
        nsamples=16,
        python="python3",
    )

    command = build_generate_act_scales_command(config)

    assert command[:2] == ["python3", str(root / "generate_act_scale_shift.py")]
    assert command[command.index("--cache_dir") + 1] == str((tmp_path / "cache").resolve())
    assert command[command.index("--scales-output-path") + 1] == str(root / "act_scales")


def test_patch_qwen25_vl_inputs_embeds_masks_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    target.write_text(
        "before\n"
        "        image_mask = None\n"
        "        text_mask =  None\n"
        "        if inputs_embeds is None:\n"
        "after\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_inputs_embeds_masks(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_inputs_embeds_masks(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: build masks for pruned inputs_embeds" in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_lmclass_attention_implementation_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "LMClass.py"
    target.write_text(
        "kwargs = {'attn_implementation': 'flash_attention_2'}\n",
        encoding="utf-8",
    )

    patch_lmclass_attention_implementation(root)
    first = target.read_text(encoding="utf-8")
    patch_lmclass_attention_implementation(root)
    second = target.read_text(encoding="utf-8")

    assert "args.attn_implementation" in first
    assert first == second


def test_patch_lmclass_qwen2_vl_support_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "LMClass.py"
    target.write_text(
        "before\n"
        "        elif 'Qwen2.5-VL' in args.model:\n"
        "            pass\n"
        "after\n",
        encoding="utf-8",
    )

    patch_lmclass_qwen2_vl_support(root)
    first = target.read_text(encoding="utf-8")
    patch_lmclass_qwen2_vl_support(root)
    second = target.read_text(encoding="utf-8")

    assert "'Qwen2-VL' in args.model" in first
    assert "Qwen2VLForConditionalGeneration" in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()
