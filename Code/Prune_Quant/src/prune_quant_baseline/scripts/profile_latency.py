import argparse
from typing import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote latency profiling skeleton.")
    parser.add_argument("--model-type", required=True, choices=["llava_onevision", "qwen2vl"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--quant-method", default="none", choices=["none", "bnb4", "bnb8", "gptq", "awq"])
    parser.add_argument("--retention-ratio", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    build_arg_parser().parse_args(argv)
    raise NotImplementedError(
        "Latency profiling is a remote-only skeleton. Record wall-clock latency, prefill/decode "
        "latency, peak GPU memory, token counts, generated tokens, quant method, and dtype here."
    )


if __name__ == "__main__":
    main()
