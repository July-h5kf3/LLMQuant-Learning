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
from vlmeval.vlm import Qwen2VLMASQuantPseudo as _PQ_Qwen2VLMASQuantPseudo
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
    pruner=_pq_os.environ.get('PQ_PRUNER', 'gae_oracle'),
    max_new_tokens=int(_pq_os.environ.get('PQ_MAX_NEW_TOKENS', '16')),
    gae_answer_source=_pq_os.environ.get('PQ_GAE_ANSWER_SOURCE', 'generated'),
    gae_per_token=_pq_bool_env('PQ_GAE_PER_TOKEN', 'false'),
    gae_quant_lambda=float(_pq_os.environ.get('PQ_GAE_QUANT_LAMBDA', '1.0')),
    gae_quant_method=_pq_os.environ.get('PQ_GAE_QUANT_METHOD', 'rtn'),
    rtn_bits=int(_pq_os.environ.get('PQ_RTN_BITS', '4')),
    rtn_group_size=int(_pq_os.environ.get('PQ_RTN_GROUP_SIZE', '0')),
    gae_score_disable_masquant_fake_quant=_pq_bool_env('PQ_GAE_DISABLE_MASQUANT_FAKE_QUANT', 'false'),
    allow_vanilla_fallback=_pq_bool_env('PQ_ALLOW_VANILLA_FALLBACK', 'false'),
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

supported_VLM['Qwen2VL_MASQuant_Pseudo'] = _pq_partial(
    _PQ_Qwen2VLMASQuantPseudo,
    model_path=_pq_os.environ.get('QWEN2VL_MODEL') or _pq_os.environ.get('QWEN25VL_MODEL', ''),
    model_type=_pq_os.environ.get('PQ_MODEL_TYPE', 'qwen2vl'),
    masquant_root=_pq_os.environ.get('MASQUANT_ROOT', ''),
    masquant_resume=_pq_os.environ.get('MASQUANT_RESUME', ''),
    masquant_act_scales=_pq_os.environ.get('MASQUANT_ACT_SCALES') or None,
    masquant_cmc_low_rank_adapters=_pq_os.environ.get('CMC_LOW_RANK') or None,
    masquant_cmc_white_matrix=_pq_os.environ.get('CMC_WHITE') or None,
    masquant_cmc_rank=float(_pq_os.environ.get('PQ_CMC_RANK', '0.2')),
    masquant_cmc_quant_cmc=int(_pq_os.environ.get('PQ_CMC_QUANT_CMC', '0')),
    masquant_wbits=int(_pq_os.environ.get('PQ_MASQUANT_WBITS', '4')),
    masquant_abits=int(_pq_os.environ.get('PQ_MASQUANT_ABITS', '8')),
    masquant_group_size=int(_pq_os.environ.get('PQ_MASQUANT_GROUP_SIZE', '0')),
    masquant_inference_mode=_pq_os.environ.get('PQ_MASQUANT_INFERENCE_MODE', 'split_scales'),
    masquant_symmetric=_pq_bool_env('PQ_MASQUANT_SYMMETRIC', 'true'),
    masquant_batch_size=int(_pq_os.environ.get('PQ_MASQUANT_BATCH_SIZE', '1')),
    retention_ratio=float(_pq_os.environ.get('PQ_RETENTION_RATIO', '1.0')),
    min_keep=int(_pq_os.environ.get('PQ_MIN_KEEP', '1')),
    pruner=_pq_os.environ.get('PQ_PRUNER', 'gae_oracle'),
    max_new_tokens=int(_pq_os.environ.get('PQ_MAX_NEW_TOKENS', '16')),
    gae_answer_source=_pq_os.environ.get('PQ_GAE_ANSWER_SOURCE', 'generated'),
    gae_per_token=_pq_bool_env('PQ_GAE_PER_TOKEN', 'false'),
    gae_quant_lambda=float(_pq_os.environ.get('PQ_GAE_QUANT_LAMBDA', '1.0')),
    gae_quant_method=_pq_os.environ.get('PQ_GAE_QUANT_METHOD', 'rtn'),
    rtn_bits=int(_pq_os.environ.get('PQ_RTN_BITS', '4')),
    rtn_group_size=int(_pq_os.environ.get('PQ_RTN_GROUP_SIZE', '0')),
    gae_score_disable_masquant_fake_quant=_pq_bool_env('PQ_GAE_DISABLE_MASQUANT_FAKE_QUANT', 'true'),
    allow_vanilla_fallback=_pq_bool_env('PQ_ALLOW_VANILLA_FALLBACK', 'false'),
    attn_implementation=_pq_os.environ.get('PQ_ATTN_IMPLEMENTATION', 'eager'),
    min_pixels=_pq_int_env('PQ_MIN_PIXELS'),
    max_pixels=_pq_int_env('PQ_MAX_PIXELS'),
    min_visual_tokens=_pq_int_env('PQ_MIN_VISUAL_TOKENS'),
    max_visual_tokens=_pq_int_env('PQ_MAX_VISUAL_TOKENS'),
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
        "from .qwen2vl_masquant_pseudo import Qwen2VLMASQuantPseudo\n"
        "from .qwen2vl_masquant_tensorrt import Qwen2VLMASQuantTensorRT\n"
        "from .qwen2vl_pruned_compressor import Qwen2VLPrunedCompressor\n"
        "from .qwen2vl_pruned_gae import Qwen2VLPrunedGAE\n",
        encoding="utf-8",
    )
    src_dir = Path(__file__).resolve().parents[1] / "src" / "prune_quant_baseline" / "vlmeval"
    shutil.copy2(src_dir / "qwen2vl_masquant_pseudo.py", target_dir / "qwen2vl_masquant_pseudo.py")
    shutil.copy2(src_dir / "qwen2vl_masquant_tensorrt.py", target_dir / "qwen2vl_masquant_tensorrt.py")
    shutil.copy2(src_dir / "qwen2vl_pruned_gae.py", target_dir / "qwen2vl_pruned_gae.py")
    shutil.copy2(src_dir / "qwen2vl_pruned_compressor.py", target_dir / "qwen2vl_pruned_compressor.py")

    init_path = vlmeval_root / "vlmeval" / "vlm" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    text = text.replace("from .prune_quant import Qwen2VLPrunedGAE\n", "")
    text = text.replace("from .prune_quant import Qwen2VLPrunedCompressor, Qwen2VLPrunedGAE\n", "")
    text = text.replace(
        "from .prune_quant import Qwen2VLMASQuantTensorRT, Qwen2VLPrunedCompressor, Qwen2VLPrunedGAE\n",
        "",
    )
    line = (
        "from .prune_quant import Qwen2VLMASQuantPseudo, Qwen2VLMASQuantTensorRT, "
        "Qwen2VLPrunedCompressor, Qwen2VLPrunedGAE\n"
    )
    if line not in text:
        init_path.write_text(text + line, encoding="utf-8")
    elif text != init_path.read_text(encoding="utf-8"):
        init_path.write_text(text, encoding="utf-8")

    _patch_config(vlmeval_root / "vlmeval" / "config.py")
    print(
        "Installed Qwen2VL_PrunedGAE, Qwen2VL_PrunedCompressor, "
        "Qwen2VL_MASQuant_Pseudo, and Qwen2VL_MASQuant_TensorRT into VLMEvalKit."
    )
    print("Set QWEN2VL_MODEL before running VLMEvalKit.")


if __name__ == "__main__":
    main()
