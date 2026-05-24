import argparse
from pathlib import Path
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a relevance compressor from oracle labels.")
    parser.add_argument("--labels-path", required=True, help="Path to JSONL/PT labels generated on remote.")
    parser.add_argument("--output-checkpoint", required=True, help="Where to save the compressor checkpoint.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    labels_path = Path(args.labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels path does not exist: {labels_path}")
    raise NotImplementedError(
        "Compressor training is a first-stage skeleton. Implement variable-length padding, "
        "KL loss against oracle distributions, and checkpoint saving before use."
    )


if __name__ == "__main__":
    main()
