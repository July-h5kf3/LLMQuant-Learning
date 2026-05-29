from pathlib import Path

import pytest

from prune_quant_baseline.quant.masquant import (
    MASQuantRunConfig,
    build_cmc_command,
    build_generate_act_scales_command,
    build_train_command,
    patch_custom_dataset_paths,
    patch_masquant_qwen2_vl_quant_support,
    patch_lmclass_attention_implementation,
    patch_lmclass_qwen2_vl_support,
    patch_qwen25_vl_linear_mask_compat,
    patch_qwen25_vl_inputs_embeds_masks,
    validate_masquant_root,
)
from prune_quant_baseline.scripts.run_prune_then_quant_masquant import build_arg_parser


def _make_masquant_root(tmp_path: Path) -> Path:
    root = tmp_path / "masquant"
    (root / "quantize").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    (root / "generate_act_scale_shift.py").write_text("", encoding="utf-8")
    (root / "quantize" / "masquant.py").write_text("", encoding="utf-8")
    (root / "models" / "LMClass.py").write_text("", encoding="utf-8")
    (root / "custom_dataset.py").write_text("", encoding="utf-8")
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


def test_build_cmc_command_uses_infer_mas_and_saves_cmc_artifacts(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    (root / "infer_mas.py").write_text("", encoding="utf-8")
    config = MASQuantRunConfig(
        masquant_root=root,
        model_path="/models/Qwen2.5-VL-7B-Instruct",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        act_scales_path=tmp_path / "act_scales" / "pruned.pt",
        python="python3",
    )

    command = build_cmc_command(
        config,
        net="qwen2.5-vl-7b",
        save_white_matrix_path=tmp_path / "cmc" / "white.pt",
        save_low_rank_adapters=tmp_path / "cmc" / "low_rank.pt",
    )

    assert command[:2] == ["python3", str(root / "infer_mas.py")]
    assert command[command.index("--mode") + 1] == "infer"
    assert command[command.index("--net") + 1] == "qwen2.5-vl-7b"
    assert command[command.index("--scales_path") + 1] == str((tmp_path / "act_scales" / "pruned.pt").resolve())
    assert command[command.index("--cache_dir") + 1] == str((tmp_path / "cache").resolve())
    assert command[command.index("--save_white_matrix_path") + 1] == str(tmp_path / "cmc" / "white.pt")
    assert command[command.index("--save_low_rank_adapters") + 1] == str(tmp_path / "cmc" / "low_rank.pt")
    assert "--LR" in command
    assert "--quantize" in command


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


def test_patch_qwen25_vl_linear_mask_compat_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    target.write_text(
        "class Qwen2_5_VLAttention:\n"
        "    def forward(self, hidden_states, attn_output, multi_modal_mask):\n"
        "        query_states = self.q_proj(hidden_states, multi_modal_mask)\n"
        "        key_states = self.k_proj(hidden_states, multi_modal_mask)\n"
        "        value_states = self.v_proj(hidden_states, multi_modal_mask)\n"
        "        attn_output = self.o_proj(attn_output, multi_modal_mask)\n"
        "\n"
        "\n"
        "def apply_multimodal_rotary_pos_emb(q, k, cos, sin):\n"
        "    pass\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_linear_mask_compat(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_linear_mask_compat(root)
    second = target.read_text(encoding="utf-8")

    assert "_prune_quant_baseline_masked_linear" in first
    assert "isinstance(module, nn.Linear)" in first
    assert "self.q_proj(hidden_states, multi_modal_mask)" not in first
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


def test_patch_masquant_qwen2_vl_quant_support_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "quantize" / "masquant.py"
    target.write_text(
        "def masquant(args, model, lm, layer, dataloader):\n"
        "    if 'Qwen2.5-Omni' in args.model:\n"
        "        use_cache = model.config.text_config.use_cache\n"
        "        model.config.text_config.use_cache = False\n"
        "    elif 'MiniCPM' in args.model or 'llama' in args.model:\n"
        "        use_cache = model.config.use_cache\n"
        "        model.config.use_cache = False\n"
        "    elif 'Qwen2.5-VL' in args.model:\n"
        "        is_llama = True\n"
        "        layers = model.language_model.layers\n"
        "        model.language_model.embed_tokens = model.language_model.embed_tokens.to(dev)\n"
        "        model.language_model.norm = model.language_model.norm.to(dev)\n"
        "        model.visual = model.visual.to(dev)\n"
        "        model.visual.rotary_pos_emb = model.visual.rotary_pos_emb.to(dev)\n"
        "        model.language_model.rotary_emb = model.language_model.rotary_emb.to(dev)\n"
        "        \n"
        "        for layer in model.language_model.layers:\n"
        "            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)\n"
        "\n"
        "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
        "\n"
        "        DecoderLayer = QuantQwenDecoderLayerV2\n"
        "        pairs = {\n"
        "            \"q_proj\":\"qkv\",\n"
        "            \"o_proj\":\"out\",\n"
        "            \"up_proj\":\"fc1\",\n"
        "            # \"down_proj\": \"fc2\"\n"
        "        }\n"
        "        layer_name_prefix = \"model.language_model.layers\"        \n"
        "    else:\n"
        "        raise ValueError(\"Only support for opt/llama/Llama-2/falcon/mixtral now\")\n"
        "    if args.epochs > 0:\n"
        "        if 'Qwen2.5-Omni' in args.model or 'Qwen2.5-VL' in args.model:\n"
        "            model(**inputs)\n"
        "    if 'Qwen2.5-VL' in args.model:\n"
        "        model.language_model.embed_tokens = model.language_model.embed_tokens.cpu()\n"
        "        model.language_model.norm = model.language_model.norm.cpu()\n"
        "    if False:\n"
        "        pass\n"
        "    elif 'Qwen2.5-VL' in args.model:\n"
        "        qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)\n",
        encoding="utf-8",
    )

    patch_masquant_qwen2_vl_quant_support(root)
    first = target.read_text(encoding="utf-8")
    patch_masquant_qwen2_vl_quant_support(root)
    second = target.read_text(encoding="utf-8")

    assert "'Qwen2-VL' in args.model" in first
    assert "layers = model.model.layers" in first
    assert "layer_name_prefix = \"model.layers\"" in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_custom_dataset_paths_replaces_upstream_hardcoded_paths(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "custom_dataset.py"
    target.write_text(
        "if data_type == 'vision-only':\n"
        "    dataset_json = '/already/patched.json'\n"
        '    prefix_path = "file:///already/patched/"\n'
        "elif data_type == 'text-vision':\n"
        "    dataset_json = '/already/patched.json'\n"
        '    prefix_path = "file:///already/patched/"\n',
        encoding="utf-8",
    )

    patch_custom_dataset_paths(
        root,
        vision_json=tmp_path / "sharegpt4v.json",
        vision_prefix=tmp_path / "coco" / "train2017",
    )
    text = target.read_text(encoding="utf-8")

    assert str((tmp_path / "sharegpt4v.json").resolve()) in text
    assert f"file://{(tmp_path / 'coco' / 'train2017').resolve()}/" in text
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_prune_then_masquant_accepts_cmc_and_export_tensorrt_stages() -> None:
    cmc_args = build_arg_parser().parse_args(
        [
            "--stage",
            "cmc",
            "--model-path",
            "/models/Qwen2.5-VL-7B-Instruct",
            "--work-dir",
            "/tmp/work",
            "--masquant-resume",
            "/tmp/mas_parameters.pth",
        ]
    )
    args = build_arg_parser().parse_args(
        [
            "--stage",
            "export-tensorrt",
            "--model-path",
            "/models/Qwen2.5-VL-7B-Instruct",
            "--work-dir",
            "/tmp/work",
            "--masquant-resume",
            "/tmp/mas_parameters.pth",
            "--tensorrt-artifact-dir",
            "/tmp/artifact",
            "--tensorrt-engine-dir",
            "/tmp/engine",
        ]
    )

    assert cmc_args.stage == "cmc"
    assert cmc_args.cmc_rank == 0.2
    assert cmc_args.cmc_cali_data_type == "vision-audio-only"
    assert args.stage == "export-tensorrt"
    assert args.tensorrt_artifact_dir == "/tmp/artifact"
    assert args.tensorrt_engine_dir == "/tmp/engine"
