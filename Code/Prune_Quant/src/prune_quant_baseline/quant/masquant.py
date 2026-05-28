from __future__ import annotations

import contextlib
import logging
import os
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


def load_masquant_model_and_processor(
    *,
    masquant_root: str | Path,
    model_id_or_path: str,
    resume: str | Path | None,
    act_scales: str | Path | None = None,
    wbits: int = 4,
    abits: int = 8,
    group_size: int | None = 0,
    symmetric: bool = True,
    inference_mode: str = "split_scales",
    attn_implementation: str = "eager",
    processor_use_fast: bool | None = None,
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
            _set_quant_params(args)
            llm = LMClass(args)
            llm.seqlen = 2048
            llm.model.eval()
            for param in llm.model.parameters():
                param.requires_grad_(False)
            loaded_act_scales = None if act_scales is None else torch.load(act_scales, weights_only=False)
            masquant(
                llm,
                args,
                _iter_no_calibration_samples(),
                loaded_act_scales,
                logging.getLogger("prune_quant_baseline.masquant"),
                None,
            )
            processor_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": local_files_only,
            }
            if processor_use_fast is not None:
                processor_kwargs["use_fast"] = processor_use_fast
            processor = AutoProcessor.from_pretrained(model_id_or_path, **processor_kwargs)
            return llm.model, processor
    finally:
        if previous_mode is None:
            os.environ.pop("inference_mode", None)
        else:
            os.environ["inference_mode"] = previous_mode
