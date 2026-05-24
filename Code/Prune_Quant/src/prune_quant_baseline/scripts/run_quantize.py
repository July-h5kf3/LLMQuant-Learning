import argparse
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantization entrypoint skeleton.")
    parser.add_argument("--model-type", required=True, choices=["llava_onevision", "qwen2vl"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--quant-method", required=True, choices=["bnb4", "bnb8", "gptq", "awq"])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    build_arg_parser().parse_args(argv)
    raise NotImplementedError("Offline quantization/export is not implemented in the first-stage baseline.")


if __name__ == "__main__":
    main()
