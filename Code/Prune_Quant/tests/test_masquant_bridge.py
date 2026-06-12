from pathlib import Path

import pytest

from prune_quant_baseline.quant.masquant import (
    MASQuantRunConfig,
    build_cmc_command,
    build_generate_act_scales_command,
    build_train_command,
    _patch_decoder_forward_compat,
    _patch_int_qwen_vl_layer_logger,
    _patch_masquant_attention_runtime_compat,
    patch_custom_dataset_paths,
    patch_masquant_qwen2_vl_quant_support,
    patch_lmclass_attention_implementation,
    patch_lmclass_qwen2_vl_support,
    patch_qwen25_vl_cmc_forward_input_compat,
    patch_qwen25_vl_linear_mask_compat,
    patch_qwen25_vl_prepare_inputs_generation_compat,
    patch_qwen25_vl_inputs_embeds_masks,
    patch_qwen25_vl_config_schema_compat,
    patch_qwen25_vl_rope_default_compat,
    validate_masquant_root,
)
from prune_quant_baseline.scripts import run_prune_then_quant_masquant
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


def _write_minimal_qwen25_modeling(root: Path) -> None:
    (root / "models" / "modeling_qwen2_5_vl.py").write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class Qwen2_5_VLRotaryEmbedding(nn.Module):\n"
        "    def __init__(self, config, device=None):\n"
        "        super().__init__()\n"
        "        if hasattr(config, \"rope_scaling\") and config.rope_scaling is not None:\n"
        "            self.rope_type = config.rope_scaling.get(\"rope_type\", config.rope_scaling.get(\"type\"))\n"
        "        else:\n"
        "            self.rope_type = \"default\"\n"
        "        self.max_seq_len_cached = config.max_position_embeddings\n"
        "        self.original_max_seq_len = config.max_position_embeddings\n"
        "        self.config = config\n"
        "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]\n"
        "        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)\n"
        "\n"
        "class Qwen2_5_VLForConditionalGeneration:\n"
        "    def __init__(self, config):\n"
        "        self.model = Qwen2_5_VLModel(config)\n"
        "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n"
        "\n"
        "    def forward(self, logits, labels):\n"
        "        image_mask = None\n"
        "        text_mask =  None\n"
        "        if inputs_embeds is None:\n"
        "            pass\n"
        "        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)\n"
        "\n"
        "    def prepare_inputs_for_generation(self, input_ids, cache_position=None, use_cache=True, **kwargs):\n"
        "        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)\n"
        "        if cache_position[0] != 0:\n"
        "            model_inputs[\"pixel_values\"] = None\n"
        "            model_inputs[\"pixel_values_videos\"] = None\n"
        "        return model_inputs\n"
        "\n"
        "def apply_multimodal_rotary_pos_emb(q, k, cos, sin):\n"
        "    pass\n",
        encoding="utf-8",
    )


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


def test_patch_qwen25_vl_rope_default_compat_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    target.write_text(
        "import torch\n"
        "\n"
        "class Qwen2_5_VLRotaryEmbedding(nn.Module):\n"
        "    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):\n"
        "        super().__init__()\n"
        "        # BC: \"rope_type\" was originally \"type\"\n"
        "        if hasattr(config, \"rope_scaling\") and config.rope_scaling is not None:\n"
        "            self.rope_type = config.rope_scaling.get(\"rope_type\", config.rope_scaling.get(\"type\"))\n"
        "        else:\n"
        "            self.rope_type = \"default\"\n"
        "        self.max_seq_len_cached = config.max_position_embeddings\n"
        "        self.original_max_seq_len = config.max_position_embeddings\n"
        "\n"
        "        self.config = config\n"
        "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]\n"
        "\n"
        "        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_rope_default_compat(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_rope_default_compat(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: support Qwen2.5-VL default RoPE" in first
    assert "def _prune_quant_baseline_default_rope_parameters" in first
    assert "self.rope_init_fn = _prune_quant_baseline_default_rope_parameters" in first
    assert "ROPE_INIT_FUNCTIONS[self.rope_type]" in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_qwen25_vl_config_schema_compat_is_idempotent(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    target.write_text(
        "def unrelated():\n"
        "    return past_key_values is None\n"
        "\n"
        "class Qwen2_5_VLForConditionalGeneration:\n"
        "    def __init__(self, config):\n"
        "        self.model = Qwen2_5_VLModel(config)\n"
        "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n"
        "\n"
        "    def forward(self, logits, labels):\n"
        "        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_config_schema_compat(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_config_schema_compat(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: support Qwen2.5-VL nested text_config" in first
    assert "config.text_config.hidden_size" in first
    assert "config.text_config.vocab_size" in first
    assert "self.config.text_config.vocab_size" in first
    assert "config.hidden_size, config.vocab_size" not in first
    assert "self.config.vocab_size" not in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_qwen25_vl_cmc_forward_input_compat_filters_unsupported_processor_keys(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "quantize" / "svd_utils.py"
    target.write_text(
        "def get_white_matrix(model, processor, multimodal_scales, args):\n"
        "    for i in range(args.n_cali_samples):\n"
        "        inputs = {k: v.to(device) for k, v in dataloader[i].items()}\n"
        "        with torch.no_grad():\n"
        "            model(**inputs)\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_cmc_forward_input_compat(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_cmc_forward_input_compat(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: filter unsupported Qwen2.5-VL CMC inputs" in first
    assert "def _prune_quant_baseline_filter_forward_inputs" in first
    assert "inputs = _prune_quant_baseline_filter_forward_inputs(model, inputs)" in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_qwen25_vl_prepare_inputs_generation_compat_handles_none_cache_position(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    target.write_text(
        "class Qwen2_5_VLForConditionalGeneration:\n"
        "    def prepare_inputs_for_generation(\n"
        "        self,\n"
        "        input_ids,\n"
        "        past_key_values=None,\n"
        "        attention_mask=None,\n"
        "        inputs_embeds=None,\n"
        "        cache_position=None,\n"
        "        position_ids=None,\n"
        "        use_cache=True,\n"
        "        pixel_values=None,\n"
        "        pixel_values_videos=None,\n"
        "        image_grid_thw=None,\n"
        "        video_grid_thw=None,\n"
        "        second_per_grid_ts=None,\n"
        "        **kwargs,\n"
        "    ):\n"
        "        model_inputs = super().prepare_inputs_for_generation(\n"
        "            input_ids,\n"
        "            cache_position=cache_position,\n"
        "            use_cache=use_cache,\n"
        "            **kwargs,\n"
        "        )\n"
        "        model_inputs[\"position_ids\"] = None\n"
        "        if cache_position[0] != 0:\n"
        "            model_inputs[\"pixel_values\"] = None\n"
        "            model_inputs[\"pixel_values_videos\"] = None\n"
        "        return model_inputs\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_prepare_inputs_generation_compat(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_prepare_inputs_generation_compat(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: tolerate missing cache_position during generation" in first
    assert "kwargs.get(\"is_first_iteration\", None)" in first
    assert "past_key_values is None" in first
    assert "cache_position is not None" in first
    assert "if cache_position[0] != 0:" not in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_qwen25_vl_prepare_inputs_generation_compat_upgrades_v1_patch(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    target.write_text(
        "class Qwen2_5_VLForConditionalGeneration:\n"
        "    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, cache_position=None, use_cache=True, **kwargs):\n"
        "        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)\n"
        "        # prune_quant_baseline: tolerate missing cache_position during generation.\n"
        "        is_first_iteration = kwargs.get(\"is_first_iteration\", None)\n"
        "        if is_first_iteration is None:\n"
        "            is_first_iteration = cache_position is None or (cache_position is not None and cache_position[0] == 0)\n"
        "        if not is_first_iteration and use_cache:\n"
        "            model_inputs[\"pixel_values\"] = None\n"
        "            model_inputs[\"pixel_values_videos\"] = None\n"
        "        return model_inputs\n",
        encoding="utf-8",
    )

    patch_qwen25_vl_prepare_inputs_generation_compat(root)
    first = target.read_text(encoding="utf-8")
    patch_qwen25_vl_prepare_inputs_generation_compat(root)
    second = target.read_text(encoding="utf-8")

    assert "past_key_values is None" in first
    assert first == second


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
        "        qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)\n"
        "    multi_modal_mask_cache = []\n"
        "    class Catcher(nn.Module):\n"
        "        def __init__(self, module):\n"
        "            super().__init__()\n"
        "            self.module = module\n"
        "            self.is_llama = False\n"
        "        def forward(self, inp, **kwargs):\n"
        "            if 'multi_modal_mask' not in kwargs and 'MiniCPM' in args.model:\n"
        "                pass\n"
        "            elif 'multi_modal_mask' in kwargs:\n"
        "                multi_modal_mask_cache.append(kwargs['multi_modal_mask'])\n",
        encoding="utf-8",
    )

    patch_masquant_qwen2_vl_quant_support(root)
    first = target.read_text(encoding="utf-8")
    patch_masquant_qwen2_vl_quant_support(root)
    second = target.read_text(encoding="utf-8")

    assert "'Qwen2-VL' in args.model" in first
    assert "qwen_language_model = getattr(model, \"language_model\", None)" in first
    assert "layers = qwen_language_model.layers" in first
    assert "qwen_layer_name_prefix = candidate" in first
    assert 'layer_name_prefix = f"{qwen_layer_name_prefix}.layers"' in first
    assert "self.attention_type = module.attention_type" in first
    assert 'cache.get("qwen2_vl_input_ids")' in first
    assert 'cache["qwen2_vl_input_ids"] = inputs.get("input_ids")' in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_int_qwen_vl_layer_logger_defines_warning_once_logger(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "int_qwen_vl_layer.py"
    target.write_text(
        "def forward_sdpa():\n"
        "    logger.warning_once('falling back')\n"
        "    return super().forward(hidden_states=hidden_states)\n",
        encoding="utf-8",
    )

    _patch_int_qwen_vl_layer_logger(root)
    first = target.read_text(encoding="utf-8")
    _patch_int_qwen_vl_layer_logger(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: define int_qwen_vl_layer logger" in first
    assert "logger = _prune_quant_baseline_hf_logging.get_logger(__name__)" in first
    assert "prune_quant_baseline: fix int_qwen_vl_layer sdpa attention fallback" in first
    assert "return _prune_quant_baseline_eager_attention_forward(self,hidden_states=hidden_states)" in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_int_qwen_vl_layer_logger_aliases_qwen25_rmsnorm(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "int_qwen_vl_layer.py"
    target.write_text(
        "from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import "
        "apply_multimodal_rotary_pos_emb, Qwen2RMSNorm\n",
        encoding="utf-8",
    )

    _patch_int_qwen_vl_layer_logger(root)
    first = target.read_text(encoding="utf-8")
    _patch_int_qwen_vl_layer_logger(root)
    second = target.read_text(encoding="utf-8")

    assert "Qwen2_5_VLRMSNorm as Qwen2RMSNorm" in first
    assert "apply_multimodal_rotary_pos_emb, Qwen2RMSNorm" not in first
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_patch_int_qwen_vl_layer_logger_unwraps_qwen25_text_config(tmp_path: Path) -> None:
    root = _make_masquant_root(tmp_path)
    target = root / "models" / "int_qwen_vl_layer.py"
    target.write_text(
        "class QuantQwenAttentionV2(nn.Module):\n"
        "    def __init__(self, org_module, config, args=None, layer_idx=0):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "        self.hidden_size = config.hidden_size\n"
        "        self.num_heads = config.num_attention_heads\n"
        "\n"
        "\n"
        "class QuantQwenDecoderLayerV2(nn.Module):\n"
        "    def __init__(self, config, ori_layer, args, layer_idx=0):\n"
        "        super().__init__()\n"
        "        self.hidden_size = config.hidden_size\n"
        "        self.self_attn = QuantQwenAttentionV2(\n"
        "            org_module=ori_layer.self_attn,\n"
        "            config=config,\n"
        "            args=args, layer_idx=layer_idx\n"
        "        )\n"
        "        self.mlp = QuantQwenMLP(\n"
        "            org_module=ori_layer.mlp,\n"
        "            hidden_size=self.hidden_size,\n"
        "            intermediate_size=config.intermediate_size,\n"
        "            hidden_act=config.hidden_act,\n"
        "            args=args,\n"
        "            layer_idx=layer_idx\n"
        "        )\n",
        encoding="utf-8",
    )

    _patch_int_qwen_vl_layer_logger(root)
    first = target.read_text(encoding="utf-8")
    _patch_int_qwen_vl_layer_logger(root)
    second = target.read_text(encoding="utf-8")

    assert "prune_quant_baseline: unwrap nested Qwen2.5-VL text config" in first
    assert "def _prune_quant_baseline_text_config(config):" in first
    assert first.count("config = _prune_quant_baseline_text_config(config)") == 2
    assert first == second
    assert target.with_suffix(target.suffix + ".prune_quant_baseline.bak").exists()


def test_qwen25_calibrate_patches_int_qwen_vl_layer_compat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_masquant_root(tmp_path)
    _write_minimal_qwen25_modeling(root)
    int_layer = root / "models" / "int_qwen_vl_layer.py"
    int_layer.write_text(
        "from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import "
        "apply_multimodal_rotary_pos_emb, Qwen2RMSNorm\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(run_prune_then_quant_masquant, "prepare_pruned_calibration_artifacts", lambda args, config: None)
    monkeypatch.setattr(run_prune_then_quant_masquant, "run_command", lambda command, *, cwd, env, dry_run=False: None)

    run_prune_then_quant_masquant.main(
        [
            "--stage",
            "calibrate",
            "--model-type",
            "qwen2_5_vl",
            "--model-path",
            "/models/Qwen2.5-VL-7B-Instruct",
            "--masquant-root",
            str(root),
            "--work-dir",
            str(tmp_path / "work"),
            "--calib-jsonl",
            str(tmp_path / "calib.jsonl"),
            "--attn-implementation",
            "flash_attention_2",
            "--patch-masquant-inputs-embeds-mask",
        ]
    )

    assert "Qwen2_5_VLRMSNorm as Qwen2RMSNorm" in int_layer.read_text(encoding="utf-8")


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


def test_patch_decoder_forward_compat_maps_transformers_cache_keyword() -> None:
    class Layer:
        attention_type = "full_attention"

        def forward(self, hidden_states, past_key_value=None, attention_mask=None):
            return hidden_states, past_key_value, attention_mask

    class LanguageModel:
        def __init__(self):
            self.layers = [Layer()]

    class Model:
        def __init__(self):
            self.model = LanguageModel()

    model = Model()

    _patch_decoder_forward_compat(model)
    result = model.model.layers[0].forward(
        "hidden",
        past_key_values="cache",
        attention_mask="mask",
        cache_position="ignored",
        position_embeddings="ignored",
    )

    assert result == ("hidden", "cache", "mask")


def test_patch_masquant_attention_runtime_compat_adds_new_transformers_attrs() -> None:
    class QuantQwenAttentionV2:
        head_dim = 128
        num_heads = 32
        num_key_value_heads = 8

    class Model:
        def __init__(self):
            self.attn = QuantQwenAttentionV2()

        def modules(self):
            return [self, self.attn]

    model = Model()
    _patch_masquant_attention_runtime_compat(model)

    assert model.attn.scaling == 128**-0.5
    assert model.attn.num_key_value_groups == 4
    assert model.attn.layer_type is None
    assert model.attn.sliding_window is None
