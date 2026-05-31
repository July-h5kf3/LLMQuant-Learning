from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence


LOGGER = logging.getLogger(__name__)
MASQUANT_SOURCE_URL = "https://github.com/alibaba/EfficientAI/tree/main/masquant"


def _append_flag(command: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def validate_masquant_root(masquant_root: str | Path | None) -> Path:
    if masquant_root is None:
        raise ValueError("MASQuant root is required. Pass --masquant-root pointing to EfficientAI/masquant.")
    root = Path(masquant_root).expanduser().resolve()
    required = [
        root / "main.py",
        root / "generate_act_scale_shift.py",
        root / "quantize" / "masquant.py",
        root / "models" / "LMClass.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "MASQuant checkout is incomplete. Expected the masquant directory from "
            f"{MASQUANT_SOURCE_URL}; missing: {', '.join(missing)}"
        )
    return root


@dataclass(frozen=True)
class MASQuantRunConfig:
    masquant_root: str | Path | None
    model_path: str
    output_dir: str | Path
    cache_dir: str | Path
    dataset_type: str = "text-vision"
    nsamples: int = 128
    batch_size: int = 1
    wbits: int = 4
    abits: int = 8
    epochs: int = 2
    group_size: int | None = 0
    symmetric: bool = True
    let: bool = True
    loss_multi_modal_mae_alpha: bool = True
    inference_mode: str = "split_scales"
    attn_implementation: str = "eager"
    act_scales_path: str | Path | None = None
    resume: str | Path | None = None
    save_dir: str | Path | None = None
    tasks_multimodal: str = ""
    eval_ppl: bool = False
    python: str = sys.executable or "python"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def root(self) -> Path:
        return validate_masquant_root(self.masquant_root)

    @property
    def net(self) -> str:
        return Path(self.model_path.rstrip("/")).name

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser().resolve()

    @property
    def cache_file(self) -> Path:
        return self.resolved_cache_dir / f"dataloader_{self.net}_{self.dataset_type}_{self.nsamples}.cache"

    @property
    def resolved_act_scales_path(self) -> Path:
        if self.act_scales_path is not None:
            return Path(self.act_scales_path).expanduser().resolve()
        return self.root / "act_scales" / f"{self.net}-{self.dataset_type}-{self.nsamples}.pt"


def build_generate_act_scales_command(config: MASQuantRunConfig) -> list[str]:
    root = config.root
    command = [
        config.python,
        str(root / "generate_act_scale_shift.py"),
        "--model",
        config.model_path,
        "--dataset-type",
        config.dataset_type,
        "--nsamples",
        str(config.nsamples),
        "--cache_dir",
        str(config.resolved_cache_dir),
        "--batch_size",
        str(config.batch_size),
        "--scales-output-path",
        str(config.resolved_act_scales_path.parent),
    ]
    return command


def build_train_command(config: MASQuantRunConfig) -> list[str]:
    root = config.root
    command = [
        config.python,
        str(root / "main.py"),
        "--model",
        config.model_path,
        "--mode",
        "train",
        "--epochs",
        str(config.epochs),
        "--wbits",
        str(config.wbits),
        "--abits",
        str(config.abits),
        "--dataset-type",
        config.dataset_type,
        "--nsamples",
        str(config.nsamples),
        "--batch_size",
        str(config.batch_size),
        "--cache_dir",
        str(config.resolved_cache_dir),
        "--output_dir",
        str(Path(config.output_dir).expanduser().resolve()),
        "--attn_implementation",
        config.attn_implementation,
        "--act-scales",
        str(config.resolved_act_scales_path),
    ]
    if config.let:
        command.append("--let")
    if config.loss_multi_modal_mae_alpha:
        command.append("--loss_multi_modal_mae_alpha")
    if config.symmetric:
        command.append("--symmetric")
    _append_flag(command, "--group_size", config.group_size)
    _append_flag(command, "--resume", config.resume)
    _append_flag(command, "--save_dir", config.save_dir)
    if config.tasks_multimodal:
        command.extend(["--tasks_multimodal", config.tasks_multimodal])
    if config.eval_ppl:
        command.append("--eval_ppl")
    command.extend(config.extra_args)
    return command


def build_cmc_command(
    config: MASQuantRunConfig,
    *,
    script_name: str = "infer_mas.py",
    net: str = "qwen2.5-vl-7b",
    scales_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_cali_samples: int = 128,
    cali_data_type: str = "vision-audio-only",
    rank: float = 0.2,
    quant_cmc: int = 0,
    save_white_matrix_path: str | Path | None = None,
    save_low_rank_adapters: str | Path | None = None,
    quantize: bool = True,
    lr: bool = True,
    tasks_multimodal: str = "",
    limit_multimodal: float | None = None,
    eval_ppl: bool = False,
    eval_sqnr: bool = False,
    eval_omni_task: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    root = config.root
    scales = Path(scales_path).expanduser().resolve() if scales_path is not None else config.resolved_act_scales_path
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else Path(config.output_dir).expanduser().resolve()
    command = [
        config.python,
        str(root / script_name),
        "--model",
        config.model_path,
        "--mode",
        "infer",
        "--net",
        net,
        "--scales_path",
        str(scales),
        "--wbits",
        str(config.wbits),
        "--abits",
        str(config.abits),
        "--output_dir",
        str(out_dir),
        "--cache_dir",
        str(config.resolved_cache_dir),
        "--batch_size",
        str(config.batch_size),
        "--n_cali_samples",
        str(n_cali_samples),
        "--cali_data_type",
        cali_data_type,
        "--rank",
        str(rank),
        "--quant_cmc",
        str(quant_cmc),
        "--attn_implementation",
        config.attn_implementation,
    ]
    if lr:
        command.append("--LR")
    if quantize:
        command.append("--quantize")
    _append_flag(command, "--save_white_matrix_path", save_white_matrix_path)
    _append_flag(command, "--save_low_rank_adapters", save_low_rank_adapters)
    if tasks_multimodal:
        command.extend(["--tasks_multimodal", tasks_multimodal])
    _append_flag(command, "--limit_multimodal", limit_multimodal)
    if eval_ppl:
        command.append("--eval_ppl")
    if eval_sqnr:
        command.append("--eval_sqnr")
    if eval_omni_task:
        command.append("--eval_omni_task")
    command.extend(extra_args)
    return command


def masquant_env(config: MASQuantRunConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["inference_mode"] = config.inference_mode
    return env


def run_command(command: Sequence[str], *, cwd: str | Path, env: dict[str, str], dry_run: bool = False) -> None:
    LOGGER.info("%s", format_command(command))
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(cwd), env=env, check=True)


def patch_qwen25_vl_inputs_embeds_masks(masquant_root: str | Path) -> Path:
    """Patch MASQuant Qwen2.5-VL to build modality masks for pruned inputs_embeds.

    The official MASQuant Qwen2.5-VL forward constructs image/text masks only when
    it creates inputs_embeds internally. Our prune-then-quant calibration cache
    intentionally passes pre-pruned inputs_embeds plus matching pruned input_ids,
    so MASQuant needs this tiny mask reconstruction before entering the language
    model. The patch is idempotent and writes a .bak file before the first edit.
    """

    root = validate_masquant_root(masquant_root)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    text = target.read_text(encoding="utf-8")
    marker = "prune_quant_baseline: build masks for pruned inputs_embeds"
    if marker in text:
        return target
    old = "        image_mask = None\n        text_mask =  None\n        if inputs_embeds is None:\n"
    new = (
        "        image_mask = None\n"
        "        text_mask =  None\n"
        "        # prune_quant_baseline: build masks for pruned inputs_embeds.\n"
        "        if inputs_embeds is not None and input_ids is not None:\n"
        "            mask = input_ids == self.config.image_token_id\n"
        "            if getattr(self.config, \"video_token_id\", None) is not None:\n"
        "                mask = mask | (input_ids == self.config.video_token_id)\n"
        "            image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)\n"
        "            all_true = torch.full(image_mask.shape, True, dtype=torch.bool, device=inputs_embeds.device)\n"
        "            text_mask = all_true & ~image_mask\n"
        "        if inputs_embeds is None:\n"
    )
    if old not in text:
        raise RuntimeError(
            f"Could not patch {target}; MASQuant source changed and the expected Qwen2.5-VL mask block was not found."
        )
    backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def patch_qwen25_vl_linear_mask_compat(masquant_root: str | Path) -> Path:
    """Patch MASQuant Qwen2.5-VL CMC whitening to tolerate native Linear layers."""

    root = validate_masquant_root(masquant_root)
    target = root / "models" / "modeling_qwen2_5_vl.py"
    text = target.read_text(encoding="utf-8")
    marker = "prune_quant_baseline: allow native Linear during CMC whitening"
    if marker in text:
        return target

    anchor = "\n\ndef apply_multimodal_rotary_pos_emb"
    helper = (
        "\n\n"
        f"# {marker}.\n"
        "def _prune_quant_baseline_masked_linear(module, x, multi_modal_mask=None):\n"
        "    if isinstance(module, nn.Linear):\n"
        "        return module(x)\n"
        "    return module(x, multi_modal_mask)\n"
    )
    replacements = {
        "self.q_proj(hidden_states, multi_modal_mask)": (
            "_prune_quant_baseline_masked_linear(self.q_proj, hidden_states, multi_modal_mask)"
        ),
        "self.k_proj(hidden_states, multi_modal_mask)": (
            "_prune_quant_baseline_masked_linear(self.k_proj, hidden_states, multi_modal_mask)"
        ),
        "self.v_proj(hidden_states, multi_modal_mask)": (
            "_prune_quant_baseline_masked_linear(self.v_proj, hidden_states, multi_modal_mask)"
        ),
        "self.o_proj(attn_output, multi_modal_mask)": (
            "_prune_quant_baseline_masked_linear(self.o_proj, attn_output, multi_modal_mask)"
        ),
    }
    missing = [old for old in replacements if old not in text]
    if anchor not in text or missing:
        raise RuntimeError(
            f"Could not patch {target}; MASQuant source changed and the expected "
            "Qwen2.5-VL masked Linear calls were not found."
        )

    patched = text.replace(anchor, helper + anchor, 1)
    for old, new in replacements.items():
        patched = patched.replace(old, new)

    backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    return target


def patch_lmclass_attention_implementation(masquant_root: str | Path) -> Path:
    """Patch MASQuant LMClass to honor args.attn_implementation.

    GAE needs eager attention during pruning. MASQuant's Qwen2.5 loaders hard-code
    flash_attention_2, which is fast but incompatible with attention-gradient
    scoring on machines where flash-attn is missing or attentions are not returned.
    """

    root = validate_masquant_root(masquant_root)
    target = root / "models" / "LMClass.py"
    text = target.read_text(encoding="utf-8")
    marker = "prune_quant_baseline: use requested attention implementation"
    if marker in text:
        return target
    old = "'attn_implementation': 'flash_attention_2'"
    if old not in text:
        raise RuntimeError(
            f"Could not patch {target}; MASQuant source changed and no hard-coded flash_attention_2 loader was found."
        )
    backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    patched = text.replace(old, "'attn_implementation': args.attn_implementation")
    target.write_text(f"# {marker}\n" + patched, encoding="utf-8")
    return target


def patch_lmclass_qwen2_vl_support(masquant_root: str | Path) -> Path:
    """Patch MASQuant LMClass to load Qwen2-VL image models.

    The upstream MASQuant checkout used by this baseline has a dedicated Qwen2.5-VL
    loader but may not include a Qwen2-VL branch. This patch inserts a Qwen2-VL
    Hugging Face loader before the Qwen2.5-VL branch, so calibration and inference
    can use the same MASQuant path for Qwen2-VL and Qwen2.5-VL.
    """

    root = validate_masquant_root(masquant_root)
    target = root / "models" / "LMClass.py"
    text = target.read_text(encoding="utf-8")
    marker = "prune_quant_baseline: add qwen2-vl loader"
    if marker in text:
        return target
    old = "        elif 'Qwen2.5-VL' in args.model:\n"
    new = (
        "        elif 'Qwen2-VL' in args.model:\n"
        "            # prune_quant_baseline: add qwen2-vl loader.\n"
        "            from transformers import Qwen2VLForConditionalGeneration\n"
        "            model_kwargs = {\n"
        "                'torch_dtype': torch.bfloat16,\n"
        "                'low_cpu_mem_usage': True,\n"
        "                'device_map': 'auto',\n"
        "                'trust_remote_code': True,\n"
        "                'attn_implementation': args.attn_implementation,\n"
        "            }\n"
        "            self.model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)\n"
        "        elif 'Qwen2.5-VL' in args.model:\n"
    )
    if old not in text:
        raise RuntimeError(
            f"Could not patch {target}; MASQuant source changed and the expected Qwen2.5-VL loader was not found."
        )
    backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def patch_masquant_qwen2_vl_quant_support(masquant_root: str | Path) -> tuple[Path, ...]:
    """Patch MASQuant quantization paths to treat Qwen2-VL as a VL model."""

    root = validate_masquant_root(masquant_root)
    patched_paths: list[Path] = []
    logger_patch = _patch_int_qwen_vl_layer_logger(root)
    if logger_patch is not None:
        patched_paths.append(logger_patch)

    target = root / "quantize" / "masquant.py"
    text = target.read_text(encoding="utf-8")
    marker = "prune_quant_baseline: add qwen2-vl quant support v2"
    if marker not in text:
        patched = text
        patched = patched.replace(
            "    elif 'MiniCPM' in args.model or 'llama' in args.model:\n",
            "    elif 'MiniCPM' in args.model or 'llama' in args.model or 'Qwen2-VL' in args.model:\n",
        )
        old_qwen2_branch = re.compile(
            r"    elif 'Qwen2-VL' in args\.model:\n(?:        .*\n|\n)+?        layer_name_prefix = \"model\.layers\"\n"
        )
        patched = old_qwen2_branch.sub("", patched, count=1)
        qwen2_branch = (
            "        layer_name_prefix = \"model.language_model.layers\"        \n"
            "    elif 'Qwen2-VL' in args.model:\n"
            "        is_llama = True\n"
            "        qwen_language_model = getattr(model, \"language_model\", None)\n"
            "        qwen_layer_name_prefix = \"language_model\"\n"
            "        if qwen_language_model is None:\n"
            "            qwen_language_model = getattr(getattr(model, \"model\", None), \"language_model\", None)\n"
            "            qwen_layer_name_prefix = \"model.language_model\"\n"
            "        if qwen_language_model is None:\n"
            "            qwen_language_model = getattr(model, \"model\", None)\n"
            "            qwen_layer_name_prefix = \"model\"\n"
            "        if qwen_language_model is None or not hasattr(qwen_language_model, \"layers\"):\n"
            "            raise AttributeError(\"Could not locate Qwen2-VL language model layers for MASQuant.\")\n"
            "        layers = qwen_language_model.layers\n"
            "        qwen_language_model.embed_tokens = qwen_language_model.embed_tokens.to(dev)\n"
            "        qwen_language_model.norm = qwen_language_model.norm.to(dev)\n"
            "        model.visual = model.visual.to(dev)\n"
            "        if hasattr(model.visual, \"rotary_pos_emb\"):\n"
            "            model.visual.rotary_pos_emb = model.visual.rotary_pos_emb.to(dev)\n"
            "        if hasattr(qwen_language_model, \"rotary_emb\"):\n"
            "            qwen_language_model.rotary_emb = qwen_language_model.rotary_emb.to(dev)\n"
            "        for layer in layers:\n"
            "            if hasattr(layer.self_attn, \"rotary_emb\"):\n"
            "                layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)\n"
            "\n"
            "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
            "\n"
            "        DecoderLayer = QuantQwenDecoderLayerV2\n"
            "        pairs = {\n"
            "            \"q_proj\":\"qkv\",\n"
            "            \"o_proj\":\"out\",\n"
            "            \"up_proj\":\"fc1\",\n"
            "        }\n"
            "        if act_scales is not None:\n"
            "            for candidate in (qwen_layer_name_prefix, \"model.language_model\", \"language_model\", \"model\"):\n"
            "                key = f\"{candidate}.layers.0.self_attn.q_proj.all_in_one_scale\"\n"
            "                if key in act_scales:\n"
            "                    qwen_layer_name_prefix = candidate\n"
            "                    break\n"
            "        layer_name_prefix = f\"{qwen_layer_name_prefix}.layers\"\n"
        )
        patched = patched.replace(
            "        layer_name_prefix = \"model.language_model.layers\"        \n",
            qwen2_branch,
            1,
        )
        patched = patched.replace(
            "                    if 'Qwen2.5-Omni' in args.model or 'Qwen2.5-VL' in args.model:\n",
            (
                "                    if 'Qwen2.5-Omni' in args.model or "
                "'Qwen2.5-VL' in args.model or 'Qwen2-VL' in args.model:\n"
            ),
        )
        patched = patched.replace(
            "            elif 'Qwen2.5-VL' in args.model:\n"
            "                qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)\n",
            "            elif 'Qwen2.5-VL' in args.model or 'Qwen2-VL' in args.model:\n"
            "                qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)\n",
        )
        patched = patched.replace(
            "        if 'Qwen2.5-VL' in args.model:\n"
            "            model.language_model.embed_tokens = model.language_model.embed_tokens.cpu()\n"
            "            model.language_model.norm = model.language_model.norm.cpu()\n"
            "        else:\n"
            "            model.model.embed_tokens = model.model.embed_tokens.cpu()\n"
            "            model.model.norm = model.model.norm.cpu()\n",
            "        if 'Qwen2.5-VL' in args.model:\n"
            "            model.language_model.embed_tokens = model.language_model.embed_tokens.cpu()\n"
            "            model.language_model.norm = model.language_model.norm.cpu()\n"
            "        elif 'Qwen2-VL' in args.model:\n"
            "            qwen_language_model.embed_tokens = qwen_language_model.embed_tokens.cpu()\n"
            "            qwen_language_model.norm = qwen_language_model.norm.cpu()\n"
            "        else:\n"
            "            model.model.embed_tokens = model.model.embed_tokens.cpu()\n"
            "            model.model.norm = model.model.norm.cpu()\n",
        )
        if patched == text:
            raise RuntimeError(
                f"Could not patch {target}; MASQuant source changed and Qwen2.5-VL quant blocks were not found."
            )
        backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        target.write_text(f"# {marker}\n" + patched, encoding="utf-8")
        patched_paths.append(target)

    text = target.read_text(encoding="utf-8")
    catcher_marker = "prune_quant_baseline: preserve qwen2-vl Catcher attention_type"
    if catcher_marker not in text:
        old = "            self.module = module\n            self.is_llama = False\n"
        new = (
            "            self.module = module\n"
            "            self.is_llama = False\n"
            f"            # {catcher_marker}.\n"
            "            if hasattr(module, \"attention_type\"):\n"
            "                self.attention_type = module.attention_type\n"
        )
        if old not in text:
            if "self.attention_type = module.attention_type" in text:
                patched = f"# {catcher_marker}\n" + text
            else:
                raise RuntimeError(f"Could not patch {target}; expected Catcher module assignment was not found.")
        else:
            patched = text.replace(old, new, 1)
        backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        target.write_text(patched, encoding="utf-8")
        if target not in patched_paths:
            patched_paths.append(target)

    text = target.read_text(encoding="utf-8")
    mask_marker = "prune_quant_baseline: cache qwen2-vl multimodal masks"
    if mask_marker not in text:
        old = (
            "            elif 'multi_modal_mask' in kwargs:\n"
            "                multi_modal_mask_cache.append(kwargs['multi_modal_mask'])\n"
        )
        new = (
            f"            # {mask_marker}.\n"
            "            elif 'Qwen2-VL' in args.model and 'Qwen2.5' not in args.model:\n"
            "                batch_size, seq_len, hidden_dim = inp.shape if inp.dim() == 3 else (1, inp.shape[0], inp.shape[1])\n"
            "                device = inp.device\n"
            "                input_ids = cache.get(\"qwen2_vl_input_ids\")\n"
            "                image_mask_2d = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)\n"
            "                if input_ids is not None:\n"
            "                    input_ids = input_ids.to(device)\n"
            "                    if input_ids.dim() == 1:\n"
            "                        input_ids = input_ids.unsqueeze(0)\n"
            "                    if input_ids.shape[0] != batch_size:\n"
            "                        input_ids = input_ids[:batch_size]\n"
            "                    if input_ids.shape[-1] < seq_len:\n"
            "                        pad = torch.full(\n"
            "                            (input_ids.shape[0], seq_len - input_ids.shape[-1]),\n"
            "                            -1,\n"
            "                            dtype=input_ids.dtype,\n"
            "                            device=device,\n"
            "                        )\n"
            "                        input_ids = torch.cat([input_ids, pad], dim=-1)\n"
            "                    elif input_ids.shape[-1] > seq_len:\n"
            "                        input_ids = input_ids[:, :seq_len]\n"
            "                    image_token_id = getattr(model.config, \"image_token_id\", None)\n"
            "                    video_token_id = getattr(model.config, \"video_token_id\", None)\n"
            "                    if image_token_id is not None:\n"
            "                        image_mask_2d |= input_ids == int(image_token_id)\n"
            "                    if video_token_id is not None:\n"
            "                        image_mask_2d |= input_ids == int(video_token_id)\n"
            "                image_mask = image_mask_2d.unsqueeze(-1).expand(batch_size, seq_len, hidden_dim)\n"
            "                all_true = torch.ones((batch_size, seq_len, hidden_dim), dtype=torch.bool, device=device)\n"
            "                text_mask = all_true & ~image_mask\n"
            "                audio_mask = None\n"
            "                multi_modal_mask_cache.append((audio_mask, image_mask, text_mask))\n"
            "                kwargs['multi_modal_mask'] = (audio_mask, image_mask, text_mask)\n"
            "            elif 'multi_modal_mask' in kwargs:\n"
            "                multi_modal_mask_cache.append(kwargs['multi_modal_mask'])\n"
        )
        if old not in text:
            if "cache.get(\"qwen2_vl_input_ids\")" in text:
                patched = f"# {mask_marker}\n" + text
            else:
                raise RuntimeError(f"Could not patch {target}; expected multimodal mask cache block was not found.")
        else:
            patched = text.replace(old, new, 1)

        old = (
            "                    if 'Qwen2.5-Omni' in args.model or 'Qwen2.5-VL' in args.model "
            "or 'Qwen2-VL' in args.model:\n"
            "                        inputs = {k: v.to(dev) for k, v in batch.items()}\n"
            "                        model(**inputs)\n"
        )
        new = (
            "                    if 'Qwen2.5-Omni' in args.model or 'Qwen2.5-VL' in args.model "
            "or 'Qwen2-VL' in args.model:\n"
            "                        inputs = {k: v.to(dev) for k, v in batch.items()}\n"
            "                        if 'Qwen2-VL' in args.model and 'Qwen2.5' not in args.model:\n"
            "                            cache[\"qwen2_vl_input_ids\"] = inputs.get(\"input_ids\")\n"
            "                        model(**inputs)\n"
        )
        if old in patched:
            patched = patched.replace(old, new, 1)
        elif "cache[\"qwen2_vl_input_ids\"] = inputs.get(\"input_ids\")" not in patched:
            raise RuntimeError(f"Could not patch {target}; expected Qwen2-VL calibration forward block was not found.")

        backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        target.write_text(patched, encoding="utf-8")
        if target not in patched_paths:
            patched_paths.append(target)

    target = root / "quantize" / "infer_quant.py"
    if target.exists():
        text = target.read_text(encoding="utf-8")
        marker = "prune_quant_baseline: add qwen2-vl infer quant support v2"
        if marker not in text:
            old = (
                "    if \"omni\" in args.model.lower():\n"
                "        layers = model.model.layers\n"
                "        from models.int_qwen_omni_layer import QuantQwenDecoderLayerV2\n"
                "    else:\n"
                "        layers = model.model.language_model.layers\n"
                "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
            )
            old_v1 = (
                "    if \"omni\" in args.model.lower():\n"
                "        layers = model.model.layers\n"
                "        from models.int_qwen_omni_layer import QuantQwenDecoderLayerV2\n"
                "    elif \"qwen2-vl\" in args.model.lower() and \"qwen2.5\" not in args.model.lower():\n"
                "        layers = model.model.layers\n"
                "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
                "    else:\n"
                "        layers = model.model.language_model.layers\n"
                "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
            )
            new = (
                "    if \"omni\" in args.model.lower():\n"
                "        layers = model.model.layers\n"
                "        from models.int_qwen_omni_layer import QuantQwenDecoderLayerV2\n"
                "    elif \"qwen2-vl\" in args.model.lower() and \"qwen2.5\" not in args.model.lower():\n"
                "        qwen_language_model = getattr(model, \"language_model\", None)\n"
                "        if qwen_language_model is None:\n"
                "            qwen_language_model = getattr(getattr(model, \"model\", None), \"language_model\", None)\n"
                "        if qwen_language_model is None:\n"
                "            qwen_language_model = getattr(model, \"model\", None)\n"
                "        if qwen_language_model is None or not hasattr(qwen_language_model, \"layers\"):\n"
                "            raise AttributeError(\"Could not locate Qwen2-VL language model layers for MASQuant.\")\n"
                "        layers = qwen_language_model.layers\n"
                "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
                "    else:\n"
                "        layers = model.model.language_model.layers\n"
                "        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2\n"
            )
            source = old_v1 if old_v1 in text else old
            if source not in text:
                if "qwen_language_model = getattr(model, \"language_model\", None)" in text:
                    new_text = f"# {marker}\n" + text
                else:
                    raise RuntimeError(f"Could not patch {target}; expected layer selection block was not found.")
            else:
                new_text = f"# {marker}\n" + text.replace(source, new, 1)
            backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
            if not backup.exists():
                backup.write_text(text, encoding="utf-8")
            target.write_text(new_text, encoding="utf-8")
            patched_paths.append(target)

    target = root / "quantize" / "svd_utils.py"
    if target.exists():
        text = target.read_text(encoding="utf-8")
        marker = "prune_quant_baseline: add qwen2-vl CMC scale prefix v2"
        if marker not in text:
            old = (
                "    if model_type == 'vl':\n"
                "        prefix = \"model.language_model.\"\n"
                "    elif model_type == 'omni':\n"
            )
            old_v1 = (
                "    if model_type == 'qwen2_vl':\n"
                "        prefix = \"model.\"\n"
                "    elif model_type == 'vl':\n"
                "        prefix = \"model.language_model.\"\n"
                "    elif model_type == 'omni':\n"
            )
            new = (
                "    if model_type == 'qwen2_vl':\n"
                "        prefix = \"model.language_model.\"\n"
                "    elif model_type == 'vl':\n"
                "        prefix = \"model.language_model.\"\n"
                "    elif model_type == 'omni':\n"
            )
            source = old_v1 if old_v1 in text else old
            if source not in text:
                if "if model_type == 'qwen2_vl':" in text:
                    new_text = f"# {marker}\n" + text
                else:
                    raise RuntimeError(f"Could not patch {target}; expected trans_scales prefix block was not found.")
            else:
                new_text = f"# {marker}\n" + text.replace(source, new, 1)
            backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
            if not backup.exists():
                backup.write_text(text, encoding="utf-8")
            target.write_text(new_text, encoding="utf-8")
            patched_paths.append(target)

    target = root / "infer_mas.py"
    if target.exists():
        text = target.read_text(encoding="utf-8")
        marker = "prune_quant_baseline: add qwen2-vl CMC entry support v2"
        if marker not in text:
            patched = text
            if '"qwen2-vl-7b"' not in patched:
                patched = patched.replace(
                    '    "qwen2.5-vl-7b",\n',
                    '    "qwen2.5-vl-7b",\n    "qwen2-vl-7b",\n',
                    1,
                )
            patched = patched.replace(
                "elif \"vl\" in args.model.lower():\n"
                "    model_type = \"vl\"\n",
                "elif \"qwen2-vl\" in args.model.lower() and \"qwen2.5\" not in args.model.lower():\n"
                "    model_type = \"qwen2_vl\"\n"
                "elif \"vl\" in args.model.lower():\n"
                "    model_type = \"vl\"\n",
                1,
            )
            patched = patched.replace(
                "elif model_type == \"qwen2_vl\":\n"
                "    down_shape = llm.model.model.layers[0].mlp.down_proj.weight.shape[1]\n",
                "elif model_type == \"qwen2_vl\":\n"
                "    qwen_language_model = getattr(llm.model, \"language_model\", None)\n"
                "    if qwen_language_model is None:\n"
                "        qwen_language_model = getattr(getattr(llm.model, \"model\", None), \"language_model\", None)\n"
                "    if qwen_language_model is None:\n"
                "        qwen_language_model = getattr(llm.model, \"model\", None)\n"
                "    if qwen_language_model is None or not hasattr(qwen_language_model, \"layers\"):\n"
                "        raise AttributeError(\"Could not locate Qwen2-VL language model layers for MASQuant.\")\n"
                "    down_shape = qwen_language_model.layers[0].mlp.down_proj.weight.shape[1]\n",
                1,
            )
            patched = patched.replace(
                "if model_type == \"omni\":\n"
                "    down_shape = llm.model.model.layers[0].mlp.down_proj.weight.shape[1]\n"
                "else:\n"
                "    down_shape = llm.model.model.language_model.layers[0].mlp.down_proj.weight.shape[1]\n",
                "if model_type == \"omni\":\n"
                "    down_shape = llm.model.model.layers[0].mlp.down_proj.weight.shape[1]\n"
                "elif model_type == \"qwen2_vl\":\n"
                "    qwen_language_model = getattr(llm.model, \"language_model\", None)\n"
                "    if qwen_language_model is None:\n"
                "        qwen_language_model = getattr(getattr(llm.model, \"model\", None), \"language_model\", None)\n"
                "    if qwen_language_model is None:\n"
                "        qwen_language_model = getattr(llm.model, \"model\", None)\n"
                "    if qwen_language_model is None or not hasattr(qwen_language_model, \"layers\"):\n"
                "        raise AttributeError(\"Could not locate Qwen2-VL language model layers for MASQuant.\")\n"
                "    down_shape = qwen_language_model.layers[0].mlp.down_proj.weight.shape[1]\n"
                "else:\n"
                "    down_shape = llm.model.model.language_model.layers[0].mlp.down_proj.weight.shape[1]\n",
                1,
            )
            if patched == text:
                raise RuntimeError(f"Could not patch {target}; expected Qwen2.5-VL CMC blocks were not found.")
            backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
            if not backup.exists():
                backup.write_text(text, encoding="utf-8")
            target.write_text(f"# {marker}\n" + patched, encoding="utf-8")
            patched_paths.append(target)

    return tuple(patched_paths)


def _patch_int_qwen_vl_layer_logger(root: Path) -> Path | None:
    """Patch MASQuant's Qwen-VL layer module for newer Transformers behavior."""

    target = root / "models" / "int_qwen_vl_layer.py"
    if not target.exists():
        return None
    text = target.read_text(encoding="utf-8")
    patched = text
    injections: list[str] = []

    logger_marker = "prune_quant_baseline: define int_qwen_vl_layer logger"
    if logger_marker not in patched and "logger.warning_once" in patched and not re.search(
        r"^logger\s*=", patched, flags=re.MULTILINE
    ):
        injections.append(
            f"# {logger_marker}.\n"
            "from transformers.utils import logging as _prune_quant_baseline_hf_logging\n"
            "logger = _prune_quant_baseline_hf_logging.get_logger(__name__)\n"
        )

    sdpa_marker = "prune_quant_baseline: fix int_qwen_vl_layer sdpa attention fallback"
    if sdpa_marker not in patched and "return super().forward(" in patched:
        injections.append(
            f"# {sdpa_marker}.\n"
            "try:\n"
            "    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention as _PQB_Qwen2VLAttention\n"
            "except Exception:\n"
            "    _PQB_Qwen2VLAttention = None\n"
            "try:\n"
            "    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention as _PQB_Qwen25VLAttention\n"
            "except Exception:\n"
            "    _PQB_Qwen25VLAttention = None\n"
            "try:\n"
            "    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention as _PQB_Qwen25OmniAttention\n"
            "except Exception:\n"
            "    _PQB_Qwen25OmniAttention = None\n"
            "\n"
            "def _prune_quant_baseline_eager_attention_forward(module, *args, **kwargs):\n"
            "    head_dim = getattr(module, 'head_dim', None)\n"
            "    if head_dim is not None and not hasattr(module, 'scaling'):\n"
            "        module.scaling = float(head_dim) ** -0.5\n"
            "    num_heads = getattr(module, 'num_heads', None)\n"
            "    num_key_value_heads = getattr(module, 'num_key_value_heads', None)\n"
            "    if num_heads is not None and num_key_value_heads not in (None, 0) and not hasattr(module, 'num_key_value_groups'):\n"
            "        module.num_key_value_groups = int(num_heads) // int(num_key_value_heads)\n"
            "    if not hasattr(module, 'layer_type'):\n"
            "        module.layer_type = None\n"
            "    if not hasattr(module, 'sliding_window'):\n"
            "        module.sliding_window = None\n"
            "    module_name = module.__class__.__name__.lower()\n"
            "    config_name = module.config.__class__.__name__.lower() if hasattr(module, 'config') else ''\n"
            "    candidates = []\n"
            "    if 'omni' in module_name or 'omni' in config_name:\n"
            "        candidates.extend([_PQB_Qwen25OmniAttention, _PQB_Qwen25VLAttention, _PQB_Qwen2VLAttention])\n"
            "    elif '2_5' in module_name or '2_5' in config_name or 'qwen2_5' in module_name or 'qwen2_5' in config_name:\n"
            "        candidates.extend([_PQB_Qwen25VLAttention, _PQB_Qwen25OmniAttention, _PQB_Qwen2VLAttention])\n"
            "    else:\n"
            "        candidates.extend([_PQB_Qwen2VLAttention, _PQB_Qwen25VLAttention, _PQB_Qwen25OmniAttention])\n"
            "    last_error = None\n"
            "    for attention_cls in candidates:\n"
            "        if attention_cls is None:\n"
            "            continue\n"
            "        try:\n"
            "            old_attn_impl = getattr(module.config, '_attn_implementation', None) if hasattr(module, 'config') else None\n"
            "            if hasattr(module, 'config'):\n"
            "                module.config._attn_implementation = 'eager'\n"
            "            try:\n"
            "                result = attention_cls.forward(module, *args, **kwargs)\n"
            "            finally:\n"
            "                if old_attn_impl is not None:\n"
            "                    module.config._attn_implementation = old_attn_impl\n"
            "            if isinstance(result, tuple) and len(result) == 2:\n"
            "                return result[0], result[1], None\n"
            "            return result\n"
            "        except TypeError as exc:\n"
            "            last_error = exc\n"
            "    if last_error is not None:\n"
            "        raise last_error\n"
            "    raise RuntimeError('No compatible Transformers eager Qwen-VL attention class is available.')\n"
        )
        patched = patched.replace(
            "return super().forward(",
            "return _prune_quant_baseline_eager_attention_forward(self,",
        )

    if patched == text and not injections:
        return None

    backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    if injections:
        patched = "\n".join(injections) + "\n\n" + patched
    target.write_text(patched, encoding="utf-8")
    return target


def patch_custom_dataset_paths(
    masquant_root: str | Path,
    *,
    vision_json: str | Path | None = None,
    vision_prefix: str | Path | None = None,
    audio_json: str | Path | None = None,
    audio_prefix: str | Path | None = None,
) -> Path:
    """Patch MASQuant custom_dataset.py away from upstream hard-coded /nas paths."""

    root = validate_masquant_root(masquant_root)
    target = root / "custom_dataset.py"
    text = target.read_text(encoding="utf-8")

    def file_prefix(path: str | Path) -> str:
        prefix = str(path)
        if not prefix.startswith("file://"):
            prefix = "file://" + str(Path(prefix).expanduser().resolve())
        if not prefix.endswith("/"):
            prefix += "/"
        return prefix

    def replace_assignment_after(branch_marker: str, variable: str, value: str, source: str) -> str:
        marker_index = source.find(branch_marker)
        if marker_index < 0:
            raise RuntimeError(f"Could not patch {target}; branch marker not found: {branch_marker}")
        pattern = re.compile(rf"(^\s*{re.escape(variable)}\s*=\s*)(['\"])(.*?)(\2)", re.MULTILINE)
        match = pattern.search(source, marker_index)
        if match is None:
            raise RuntimeError(f"Could not patch {target}; assignment not found after {branch_marker}: {variable}")
        return source[: match.start()] + f"{match.group(1)}{match.group(2)}{value}{match.group(2)}" + source[match.end() :]

    patched = text
    if vision_json is not None:
        value = str(Path(vision_json).expanduser().resolve())
        patched = replace_assignment_after("data_type == 'vision-only'", "dataset_json", value, patched)
        patched = replace_assignment_after("data_type == 'text-vision'", "dataset_json", value, patched)
    if vision_prefix is not None:
        value = file_prefix(vision_prefix)
        patched = replace_assignment_after("data_type == 'vision-only'", "prefix_path", value, patched)
        patched = replace_assignment_after("data_type == 'text-vision'", "prefix_path", value, patched)
    if audio_json is not None:
        value = str(Path(audio_json).expanduser().resolve())
        patched = replace_assignment_after("data_type == 'audio-only'", "dataset_json", value, patched)
        patched = replace_assignment_after("data_type == 'text-audio'", "dataset_json", value, patched)
    if audio_prefix is not None:
        value = file_prefix(audio_prefix)
        patched = replace_assignment_after("data_type == 'audio-only'", "prefix_path", value, patched)

    if patched == text:
        return target
    backup = target.with_suffix(target.suffix + ".prune_quant_baseline.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    return target


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    old_path = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old_path


def _masquant_namespace(
    *,
    model_id_or_path: str,
    resume: str | Path | None,
    act_scales: str | Path | None,
    wbits: int,
    abits: int,
    group_size: int | None,
    symmetric: bool,
    attn_implementation: str,
    batch_size: int,
) -> SimpleNamespace:
    net = Path(model_id_or_path.rstrip("/")).name
    return SimpleNamespace(
        model=model_id_or_path,
        mode="train",
        cache_dir="./cache",
        output_dir="",
        output_dir_postfix="",
        save_dir=None,
        resume=str(resume) if resume is not None else None,
        calib_dataset="omnibench",
        nsamples=0,
        batch_size=batch_size,
        seed=2,
        tasks="",
        tasks_multimodal="",
        eval_ppl=False,
        auto_scale=False,
        auto_alpha=False,
        auto_epochs=False,
        loss_multi_modal=False,
        loss_multi_modal_mae=False,
        loss_multi_modal_mae_alpha=False,
        ppl_result="ppl_result.csv",
        eval_sqnr=False,
        sqnr_result="sqnr_result.csv",
        num_fewshot=0,
        wbits=wbits,
        abits=abits,
        group_size=group_size,
        alpha=0.5,
        let_lr=5e-2,
        lwc_lr=1e-2,
        wd=0,
        epochs=0,
        let=True,
        lwc=False,
        aug_loss=False,
        symmetric=symmetric,
        disable_zero_point=False,
        a_dynamic_method="per_token",
        w_dynamic_method="per_channel",
        limit=-1,
        limit_multimodal=1.0,
        multigpu=False,
        deactive_amp=abits < 16 or wbits < 16,
        attn_implementation=attn_implementation,
        net=net,
        act_scales=str(act_scales) if act_scales is not None else None,
        act_shifts=None,
        input_file="",
        output_file="",
        grad_info_path="",
        eval_omni_task=False,
        dataset_type="text-vision",
        model_family=net.split("-")[0],
    )


def _set_quant_params(args: SimpleNamespace) -> None:
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = dict(args.q_quant_params)
    args.v_quant_params = dict(args.q_quant_params)
    args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}


def _iter_no_calibration_samples() -> Iterable[Any]:
    return ()


def _language_model_layers(model: Any) -> Any:
    for candidate in (
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
    ):
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise AttributeError("Could not locate Qwen2-VL language model layers for MASQuant.")


def _capture_attention_types(layers: Any) -> list[Any]:
    return [getattr(layer, "attention_type", None) for layer in layers]


def _restore_attention_types(model: Any, attention_types: Sequence[Any]) -> None:
    for layer, attention_type in zip(_language_model_layers(model), attention_types):
        if attention_type is not None and not hasattr(layer, "attention_type"):
            layer.attention_type = attention_type


def _patch_decoder_forward_compat(model: Any) -> None:
    import inspect
    from types import MethodType

    for layer in _language_model_layers(model):
        if getattr(layer, "_prune_quant_baseline_forward_compat", False):
            continue
        original_forward = layer.forward
        try:
            signature = inspect.signature(original_forward)
        except (TypeError, ValueError):
            continue
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            continue
        accepted = set(signature.parameters)

        def compat_forward(self: Any, *args: Any, __forward: Any = original_forward, __accepted: set[str] = accepted, **kwargs: Any) -> Any:
            del self
            if "past_key_values" in kwargs and "past_key_values" not in __accepted:
                value = kwargs.pop("past_key_values")
                if "past_key_value" in __accepted:
                    kwargs["past_key_value"] = value
            for key in ("cache_position", "position_embeddings"):
                if key not in __accepted:
                    kwargs.pop(key, None)
            kwargs = {key: value for key, value in kwargs.items() if key in __accepted}
            return __forward(*args, **kwargs)

        layer.forward = MethodType(compat_forward, layer)
        layer._prune_quant_baseline_forward_compat = True


def _patch_masquant_attention_runtime_compat(model: Any) -> None:
    for module in model.modules():
        class_name = module.__class__.__name__
        if "Attention" not in class_name:
            continue
        head_dim = getattr(module, "head_dim", None)
        if head_dim is not None and not hasattr(module, "scaling"):
            module.scaling = float(head_dim) ** -0.5
        num_heads = getattr(module, "num_heads", None)
        num_key_value_heads = getattr(module, "num_key_value_heads", None)
        if (
            num_heads is not None
            and num_key_value_heads not in (None, 0)
            and not hasattr(module, "num_key_value_groups")
        ):
            module.num_key_value_groups = int(num_heads) // int(num_key_value_heads)
        if not hasattr(module, "layer_type"):
            module.layer_type = None
        if not hasattr(module, "sliding_window"):
            module.sliding_window = None


def _masquant_cmc_model_type(model_type: str, model_id_or_path: str) -> str:
    if model_type == "qwen2vl" or ("qwen2-vl" in model_id_or_path.lower() and "qwen2.5" not in model_id_or_path.lower()):
        return "qwen2_vl"
    return "vl"


def _apply_cmc_quantization(
    *,
    model: Any,
    masquant_root: Path,
    model_type: str,
    model_id_or_path: str,
    low_rank_adapters_path: str | Path,
    args: SimpleNamespace,
) -> Any:
    import torch

    with _prepend_sys_path(masquant_root):
        from quantize.infer_quant import mas_quantize_model
        from quantize.svd_utils import trans_scales

        scales = torch.load(args.scales_path, weights_only=False)
        low_rank_adapters = torch.load(low_rank_adapters_path, weights_only=False)
        layers = _language_model_layers(model)
        attention_types = _capture_attention_types(layers)
        down_shape = layers[0].mlp.down_proj.weight.shape[1]
        text_scales, vision_scales, audio_scales = trans_scales(
            scales,
            down_shape,
            _masquant_cmc_model_type(model_type, model_id_or_path),
        )
        quantized_model = mas_quantize_model(
            model,
            low_rank_adapters=low_rank_adapters,
            text_scales=text_scales,
            vision_scales=vision_scales,
            audio_scales=audio_scales,
            args=args,
        )
        _restore_attention_types(quantized_model, attention_types)
        _patch_decoder_forward_compat(quantized_model)
        return quantized_model


def load_masquant_model_and_processor(
    *,
    masquant_root: str | Path,
    model_id_or_path: str,
    resume: str | Path | None,
    act_scales: str | Path | None = None,
    model_type: str = "qwen2vl",
    cmc_low_rank_adapters: str | Path | None = None,
    cmc_white_matrix: str | Path | None = None,
    cmc_rank: float = 0.2,
    cmc_quant_cmc: int = 0,
    wbits: int = 4,
    abits: int = 8,
    group_size: int | None = 0,
    symmetric: bool = True,
    inference_mode: str = "split_scales",
    attn_implementation: str = "eager",
    processor_use_fast: bool | None = None,
    processor_min_pixels: int | None = None,
    processor_max_pixels: int | None = None,
    local_files_only: bool = True,
    batch_size: int = 1,
) -> tuple[Any, Any]:
    """Load a MASQuant-quantized model for the existing pruned inference path.

    MASQuant lives as a separate research checkout. This loader imports that checkout
    in-process, applies saved MAS parameters with zero calibration epochs, and returns
    the transformed model plus a Hugging Face processor. The pruning step is still
    handled by this repository's GAE pipeline after the quantized model is loaded.
    """

    if resume is None and act_scales is None:
        raise ValueError("MASQuant inference requires --masquant-resume or --masquant-act-scales.")
    root = validate_masquant_root(masquant_root)
    patch_lmclass_qwen2_vl_support(root)
    patch_masquant_qwen2_vl_quant_support(root)
    if cmc_low_rank_adapters is not None:
        cmc_path = Path(cmc_low_rank_adapters).expanduser().resolve()
        if not cmc_path.exists():
            raise FileNotFoundError(cmc_path)
        if resume is None:
            raise ValueError("CMC pseudo quant loading requires --masquant-resume pointing to mas_parameters.pth.")
        if cmc_white_matrix is not None:
            white_path = Path(cmc_white_matrix).expanduser().resolve()
            if not white_path.exists():
                raise FileNotFoundError(white_path)
    if attn_implementation != "flash_attention_2":
        patch_lmclass_attention_implementation(root)
    previous_mode = os.environ.get("inference_mode")
    os.environ["inference_mode"] = inference_mode
    try:
        with _prepend_sys_path(root):
            import torch
            from models.LMClass import LMClass
            from quantize.masquant import masquant
            from transformers import AutoProcessor

            args = _masquant_namespace(
                model_id_or_path=model_id_or_path,
                resume=resume,
                act_scales=act_scales,
                wbits=wbits,
                abits=abits,
                group_size=group_size,
                symmetric=symmetric,
                attn_implementation=attn_implementation,
                batch_size=batch_size,
            )
            args.mode = "infer" if cmc_low_rank_adapters is not None else "train"
            args.rank = float(cmc_rank)
            args.quant_cmc = int(cmc_quant_cmc)
            args.scales_path = str(resume) if resume is not None else None
            _set_quant_params(args)
            llm = LMClass(args)
            llm.seqlen = 2048
            llm.model.eval()
            for param in llm.model.parameters():
                param.requires_grad_(False)
            attention_types = _capture_attention_types(_language_model_layers(llm.model))
            if cmc_low_rank_adapters is None:
                loaded_act_scales = None if act_scales is None else torch.load(act_scales, weights_only=False)
                masquant(
                    llm,
                    args,
                    _iter_no_calibration_samples(),
                    loaded_act_scales,
                    logging.getLogger("prune_quant_baseline.masquant"),
                    None,
                )
                model = llm.model
                _restore_attention_types(model, attention_types)
                _patch_decoder_forward_compat(model)
                _patch_masquant_attention_runtime_compat(model)
            else:
                if torch.cuda.is_available():
                    llm.model.to("cuda")
                model = _apply_cmc_quantization(
                    model=llm.model,
                    masquant_root=root,
                    model_type=model_type,
                    model_id_or_path=model_id_or_path,
                    low_rank_adapters_path=cmc_low_rank_adapters,
                    args=args,
                )
                _patch_masquant_attention_runtime_compat(model)
            processor_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": local_files_only,
            }
            if processor_use_fast is not None:
                processor_kwargs["use_fast"] = processor_use_fast
            if processor_min_pixels is not None:
                processor_kwargs["min_pixels"] = int(processor_min_pixels)
            if processor_max_pixels is not None:
                processor_kwargs["max_pixels"] = int(processor_max_pixels)
            processor = AutoProcessor.from_pretrained(model_id_or_path, **processor_kwargs)
            return model, processor
    finally:
        if previous_mode is None:
            os.environ.pop("inference_mode", None)
        else:
            os.environ["inference_mode"] = previous_mode
