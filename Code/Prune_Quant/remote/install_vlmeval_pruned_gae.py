from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Prune_Quant VLMEvalKit model wrappers.")
    parser.add_argument("--vlmeval-root", required=True)
    args = parser.parse_args()

    vlmeval_root = Path(args.vlmeval_root).resolve()
    target_dir = vlmeval_root / "vlmeval" / "vlm" / "prune_quant"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "__init__.py").write_text(
        "from .qwen2vl_pruned_compressor import Qwen2VLPrunedCompressor\n"
        "from .qwen2vl_pruned_gae import Qwen2VLPrunedGAE\n",
        encoding="utf-8",
    )
    src_dir = Path(__file__).resolve().parents[1] / "src" / "prune_quant_baseline" / "vlmeval"
    shutil.copy2(src_dir / "qwen2vl_pruned_gae.py", target_dir / "qwen2vl_pruned_gae.py")
    shutil.copy2(src_dir / "qwen2vl_pruned_compressor.py", target_dir / "qwen2vl_pruned_compressor.py")

    init_path = vlmeval_root / "vlmeval" / "vlm" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    line = "from .prune_quant import Qwen2VLPrunedCompressor, Qwen2VLPrunedGAE\n"
    if line not in text:
        init_path.write_text(text + line, encoding="utf-8")


if __name__ == "__main__":
    main()
