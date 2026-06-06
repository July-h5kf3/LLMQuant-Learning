#!/usr/bin/env python3
"""Probe whether lmms-eval Hugging Face datasets can load from local cache."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_HF_HOME = Path("/home/aistudio/data/datasets/387822/abcd/hf_home")
DEFAULT_DATASETS = (
    ("lmms-lab/MMMU", None, "validation"),
    ("echo840/OCRBench", None, "test"),
    ("lmms-lab/VizWiz-VQA", None, "val"),
    ("lmms-lab/ScienceQA", "ScienceQA-IMG", "test"),
    ("lmms-lab/textvqa", None, "validation"),
)


def _hub_cache_dir(hf_home: Path, repo_id: str) -> Path:
    return hf_home / "hub" / f"datasets--{repo_id.replace('/', '--')}"


def _find_broken_symlinks(root: Path) -> list[Path]:
    if not root.exists():
        return []
    broken: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() and not path.exists():
            broken.append(path)
    return broken


def _print_path_state(label: str, path: Path) -> None:
    print(f"{label}: {path} exists={path.exists()} is_dir={path.is_dir()}")


def _set_hf_env(hf_home: Path) -> None:
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_MODULES_CACHE"] = str(hf_home / "modules")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def _probe_dataset(repo_id: str, name: str | None, split: str, limit: int) -> bool:
    import datasets

    print(f"\n[probe] load_dataset path={repo_id} name={name or '<none>'} split={split}")
    try:
        ds = datasets.load_dataset(
            repo_id,
            name=name,
            split=split,
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
            download_config=datasets.DownloadConfig(local_files_only=True),
            num_proc=1,
        )
        print(f"[probe] ok rows={len(ds)} columns={list(ds.column_names)}")
        if limit > 0:
            _ = ds[: min(limit, len(ds))]
        return True
    except Exception as exc:
        print(f"[probe] failed: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    hf_home = args.hf_home.expanduser().resolve()
    _set_hf_env(hf_home)
    print("[probe] Hugging Face cache environment")
    for name in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        print(f"{name}={os.environ[name]}")

    _print_path_state("hf_home", hf_home)
    _print_path_state("datasets", hf_home / "datasets")
    _print_path_state("hub", hf_home / "hub")
    _print_path_state("modules", hf_home / "modules")

    targets = []
    if args.dataset:
        for item in args.dataset:
            parts = item.split(":", 2)
            repo_id = parts[0]
            name = parts[1] or None if len(parts) > 1 else None
            split = parts[2] if len(parts) > 2 else "validation"
            targets.append((repo_id, name, split))
    else:
        targets = list(DEFAULT_DATASETS)

    for repo_id, _, _ in targets:
        repo_cache = _hub_cache_dir(hf_home, repo_id)
        _print_path_state(f"hub[{repo_id}]", repo_cache)
        _print_path_state(f"hub[{repo_id}]/refs/main", repo_cache / "refs" / "main")
        _print_path_state(f"hub[{repo_id}]/snapshots", repo_cache / "snapshots")

    broken_links = _find_broken_symlinks(hf_home / "hub")
    if broken_links:
        print("\n[probe] broken symlinks under hub:")
        for path in broken_links[:50]:
            print(f"  {path} -> {os.readlink(path)}")
        if len(broken_links) > 50:
            print(f"  ... {len(broken_links) - 50} more")

    try:
        import datasets

        print(f"\n[probe] datasets version={datasets.__version__}")
    except Exception as exc:
        print(f"\n[probe] cannot import datasets: {type(exc).__name__}: {exc}")
        return 1

    ok = True
    for repo_id, name, split in targets:
        ok = _probe_dataset(repo_id, name, split, args.limit) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
