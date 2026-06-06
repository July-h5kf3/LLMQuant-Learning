from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "remote" / "check_lmms_hf_cache.py"
    spec = importlib.util.spec_from_file_location("check_lmms_hf_cache", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hub_cache_dir_uses_huggingface_repo_cache_name(tmp_path: Path) -> None:
    module = _load_probe_module()

    assert module._hub_cache_dir(tmp_path, "lmms-lab/MMMU") == tmp_path / "hub" / "datasets--lmms-lab--MMMU"


def test_find_broken_symlinks_reports_only_dangling_links(tmp_path: Path) -> None:
    module = _load_probe_module()
    real_file = tmp_path / "real.txt"
    real_file.write_text("ok", encoding="utf-8")
    good_link = tmp_path / "good"
    good_link.symlink_to(real_file)
    bad_link = tmp_path / "bad"
    bad_link.symlink_to(tmp_path / "missing")

    broken = module._find_broken_symlinks(tmp_path)

    assert broken == [bad_link]
