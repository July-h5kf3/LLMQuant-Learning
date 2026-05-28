from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CONFIG_BLOCK_BEGIN = "# Prune_Quant custom models BEGIN"
CONFIG_BLOCK_END = "# Prune_Quant custom models END"


def _patch_config(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    block = f"""
{CONFIG_BLOCK_BEGIN}
import os as _pq_os
from functools import partial as _pq_partial
from vlmeval.vlm import Qwen2VLMASQuantTensorRT as _PQ_Qwen2VLMASQuantTensorRT
from vlmeval.vlm import Qwen2VLPrunedCompressor as _PQ_Qwen2VLPrunedCompressor
from vlmeval.vlm import Qwen2VLPrunedGAE as _PQ_Qwen2VLPrunedGAE


def _pq_bool_env(name, default):
    value = _pq_os.environ.get(name, default)
    return str(value).strip().lower() in {{'1', 'true', 'yes', 'y'}}


def _pq_int_env(name, default=None):
    value = _pq_os.environ.get(name)
    if value is None or value == '':
        return default
    return int(value)


supported_VLM['Qwen2VL_PrunedGAE'] = _pq_partial(
    _PQ_Qwen2VLPrunedGAE,
    model_path=_pq_os.environ.get('QWEN2VL_MODEL', ''),
    retention_ratio=float(_pq_os.environ.get('PQ_RETENTION_RATIO', '0.5')),
    min_keep=int(_pq_os.environ.get('PQ_MIN_KEEP', '1')),
    max_new_tokens=int(_pq_os.environ.get('PQ_MAX_NEW_TOKENS', '16')),
    gae_answer_source=_pq_os.environ.get('PQ_GAE_ANSWER_SOURCE', 'generated'),
    gae_per_token=_pq_bool_env('PQ_GAE_PER_TOKEN', 'false'),
    attn_implementation=_pq_os.environ.get('PQ_ATTN_IMPLEMENTATION', 'eager'),
    min_pixels=_pq_int_env('PQ_MIN_PIXELS'),
    max_pixels=_pq_int_env('PQ_MAX_PIXELS'),
    min_visual_tokens=_pq_int_env('PQ_MIN_VISUAL_TOKENS'),
    max_visual_tokens=_pq_int_env('PQ_MAX_VISUAL_TOKENS'),
)

supported_VLM['Qwen2VL_PrunedCompressor'] = _pq_partial(
    _PQ_Qwen2VLPrunedCompressor,
    model_path=_pq_os.environ.get('QWEN2VL_MODEL', ''),
    retention_ratio=float(_pq_os.environ.get('PQ_RETENTION_RATIO', '0.5')),
    min_keep=int(_pq_os.environ.get('PQ_MIN_KEEP', '1')),
    max_new_tokens=int(_pq_os.environ.get('PQ_MAX_NEW_TOKENS', '16')),
    attn_implementation=_pq_os.environ.get('PQ_ATTN_IMPLEMENTATION', 'eager'),
    min_pixels=_pq_int_env('PQ_MIN_PIXELS'),
    max_pixels=_pq_int_env('PQ_MAX_PIXELS'),
    min_visual_tokens=_pq_int_env('PQ_MIN_VISUAL_TOKENS'),
    max_visual_tokens=_pq_int_env('PQ_MAX_VISUAL_TOKENS'),
    compressor_checkpoint=_pq_os.environ.get('PQ_COMPRESSOR_CHECKPOINT', ''),
)

supported_VLM['Qwen2VL_MASQuant_TensorRT'] = _pq_partial(
    _PQ_Qwen2VLMASQuantTensorRT,
    artifact_dir=_pq_os.environ.get('PQ_MASQUANT_TRT_ARTIFACT', ''),
    model_path=_pq_os.environ.get('QWEN25VL_MODEL') or None,
    max_new_tokens=int(_pq_os.environ.get('PQ_MAX_NEW_TOKENS', '16')),
    min_pixels=_pq_int_env('PQ_MIN_PIXELS'),
    max_pixels=_pq_int_env('PQ_MAX_PIXELS'),
    min_visual_tokens=_pq_int_env('PQ_MIN_VISUAL_TOKENS'),
    max_visual_tokens=_pq_int_env('PQ_MAX_VISUAL_TOKENS'),
    runtime_class=_pq_os.environ.get('PQ_TRT_RUNTIME_CLASS') or None,
)
{CONFIG_BLOCK_END}
""".strip()
    if CONFIG_BLOCK_BEGIN in text and CONFIG_BLOCK_END in text:
        before = text.split(CONFIG_BLOCK_BEGIN, 1)[0].rstrip()
        after = text.split(CONFIG_BLOCK_END, 1)[1].lstrip()
        text = f"{before}\n\n{block}\n\n{after}"
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    config_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Prune_Quant VLMEvalKit model wrappers.")
    parser.add_argument("--vlmeval-root", required=True)
    args = parser.parse_args()

    vlmeval_root = Path(args.vlmeval_root).resolve()
    target_dir = vlmeval_root / "vlmeval" / "vlm" / "prune_quant"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "__init__.py").write_text(
        "from .qwen2vl_masquant_tensorrt import Qwen2VLMASQuantTensorRT\n"
        "from .qwen2vl_pruned_compressor import Qwen2VLPrunedCompressor\n"
        "from .qwen2vl_pruned_gae import Qwen2VLPrunedGAE\n",
        encoding="utf-8",
    )
    src_dir = Path(__file__).resolve().parents[1] / "src" / "prune_quant_baseline" / "vlmeval"
    shutil.copy2(src_dir / "qwen2vl_masquant_tensorrt.py", target_dir / "qwen2vl_masquant_tensorrt.py")
    shutil.copy2(src_dir / "qwen2vl_pruned_gae.py", target_dir / "qwen2vl_pruned_gae.py")
    shutil.copy2(src_dir / "qwen2vl_pruned_compressor.py", target_dir / "qwen2vl_pruned_compressor.py")

    init_path = vlmeval_root / "vlmeval" / "vlm" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    text = text.replace("from .prune_quant import Qwen2VLPrunedGAE\n", "")
    text = text.replace("from .prune_quant import Qwen2VLPrunedCompressor, Qwen2VLPrunedGAE\n", "")
    line = "from .prune_quant import Qwen2VLMASQuantTensorRT, Qwen2VLPrunedCompressor, Qwen2VLPrunedGAE\n"
    if line not in text:
        init_path.write_text(text + line, encoding="utf-8")
    elif text != init_path.read_text(encoding="utf-8"):
        init_path.write_text(text, encoding="utf-8")

    _patch_config(vlmeval_root / "vlmeval" / "config.py")
    print("Installed Qwen2VL_PrunedGAE, Qwen2VL_PrunedCompressor, and Qwen2VL_MASQuant_TensorRT into VLMEvalKit.")
    print("Set QWEN2VL_MODEL before running VLMEvalKit.")


if __name__ == "__main__":
    main()
