#!/usr/bin/env python3
"""Run VLMEvalKit with safer judging and result reuse defaults."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_EXACT_MATCH_DATASETS = ("MME", "MMStar")
OPENAI_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_MODEL",
    "OPENAI_API_TYPE",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    "LOCAL_LLM",
)


def _split_dataset_list(value: str) -> set[str]:
    return {item.strip() for item in value.replace(",", " ").split() if item.strip()}


def _model_dir_name(model: str) -> str:
    return model.replace("/", "--")


def _find_prediction_xlsx(work_dir: Path, model: str, dataset: str) -> list[Path]:
    model_dir = work_dir / _model_dir_name(model)
    pattern = f"{model}_{dataset}.xlsx"
    if model_dir.exists():
        hits = sorted(model_dir.rglob(pattern))
        if hits:
            return hits
    return sorted(work_dir.rglob(pattern))


def _find_score_files(work_dir: Path, model: str, dataset: str) -> list[Path]:
    model_dir = work_dir / _model_dir_name(model)
    prefix = f"{model}_{dataset}_"
    suffixes = (".csv", ".json", ".xlsx")
    roots = [model_dir] if model_dir.exists() else []
    roots.append(work_dir)
    hits: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if not path.name.startswith(prefix):
                continue
            if any(marker in path.stem for marker in ("_score", "_acc", "_result")):
                hits.append(path)
    return sorted(set(hits))


def _build_run_command(
    *,
    python: str,
    run_py: Path,
    dataset: str,
    model: str,
    work_dir: Path,
    mode: str,
    judge: str | None,
    verbose: bool,
    reuse: bool,
    reuse_aux: bool,
) -> list[str]:
    cmd = [
        python,
        str(run_py),
        "--data",
        dataset,
        "--model",
        model,
        "--work-dir",
        str(work_dir),
        "--mode",
        mode,
    ]
    if judge:
        cmd += ["--judge", judge]
    if verbose:
        cmd.append("--verbose")
    if reuse:
        cmd.append("--reuse")
    if reuse_aux:
        cmd += ["--reuse-aux", "all"]
    return cmd


def _env_for_dataset(dataset: str, exact_match_datasets: set[str]) -> dict[str, str]:
    env = os.environ.copy()
    if dataset in exact_match_datasets:
        for name in OPENAI_ENV_VARS:
            env.pop(name, None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlmeval-root", required=True, type=Path)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mode", choices=("auto", "all", "infer", "eval"), default="auto")
    parser.add_argument("--judge", default="")
    parser.add_argument("--exact-match-datasets", default=" ".join(DEFAULT_EXACT_MATCH_DATASETS))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--no-reuse-aux", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    vlmeval_root = args.vlmeval_root.resolve()
    run_py = vlmeval_root / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"VLMEvalKit run.py not found: {run_py}")

    work_dir = args.work_dir.resolve()
    exact_match_datasets = _split_dataset_list(args.exact_match_datasets)
    reuse = not args.no_reuse
    reuse_aux = not args.no_reuse_aux
    failures: list[str] = []

    for dataset in args.data:
        score_files = _find_score_files(work_dir, args.model, dataset)
        prediction_files = _find_prediction_xlsx(work_dir, args.model, dataset)

        if score_files and not args.force_eval:
            print(f"[vlmeval-smart] {dataset}: score already exists, skipping.", flush=True)
            for path in score_files:
                print(f"[vlmeval-smart]   {path}", flush=True)
            continue

        if args.mode == "auto":
            mode = "eval" if prediction_files else "all"
        else:
            mode = args.mode

        judge = "exact_matching" if dataset in exact_match_datasets else (args.judge or None)
        print(
            f"[vlmeval-smart] {dataset}: mode={mode}, "
            f"judge={judge or 'VLMEvalKit default'}, reuse={reuse}",
            flush=True,
        )
        if prediction_files and mode == "eval":
            for path in prediction_files:
                print(f"[vlmeval-smart]   reusing prediction: {path}", flush=True)

        cmd = _build_run_command(
            python=args.python,
            run_py=run_py,
            dataset=dataset,
            model=args.model,
            work_dir=work_dir,
            mode=mode,
            judge=judge,
            verbose=args.verbose,
            reuse=reuse,
            reuse_aux=reuse_aux,
        )
        result = subprocess.run(
            cmd,
            cwd=vlmeval_root,
            env=_env_for_dataset(dataset, exact_match_datasets),
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"{dataset}: VLMEvalKit exited with {result.returncode}")
            continue

        score_files = _find_score_files(work_dir, args.model, dataset)
        if score_files:
            print(f"[vlmeval-smart] {dataset}: score files:", flush=True)
            for path in score_files:
                print(f"[vlmeval-smart]   {path}", flush=True)
        else:
            failures.append(f"{dataset}: no score file was produced")

    if failures:
        print("[vlmeval-smart] Some datasets did not produce scores:", file=sys.stderr)
        for item in failures:
            print(f"[vlmeval-smart]   {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
