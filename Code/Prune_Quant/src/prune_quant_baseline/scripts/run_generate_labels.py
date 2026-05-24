import argparse
from typing import Sequence

from prune_quant_baseline.core.logging_utils import configure_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GAE oracle labels on a remote machine.")
    parser.add_argument("--model-type", required=True, choices=["llava_onevision", "qwen2vl"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)
    raise NotImplementedError("GAE label generation is a stage-2 skeleton.")


if __name__ == "__main__":
    main()
