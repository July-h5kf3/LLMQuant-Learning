from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "remote" / "run_lmms_eval_smart.py"
    spec = importlib.util.spec_from_file_location("run_lmms_eval_smart", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lmms_eval_default_tasks_match_metric_plan() -> None:
    module = _load_runner_module()

    assert module.DEFAULT_TASKS == ("mmmu_val", "ocrbench", "vizwiz_vqa_val", "scienceqa_img", "textvqa_val")


def test_lmms_eval_model_args_from_environment(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setenv("QWEN2VL_MODEL", "/models/qwen2vl")
    monkeypatch.setenv("PQ_RETENTION_RATIO", "1.0")
    monkeypatch.setenv("PQ_MAX_VISUAL_TOKENS", "1500")
    monkeypatch.delenv("PQ_MIN_VISUAL_TOKENS", raising=False)

    args = Namespace(model_args="", model_path="")
    model_args = module._build_default_model_args(args)

    assert "pretrained=/models/qwen2vl" in model_args
    assert "retention_ratio=1.0" in model_args
    assert "max_visual_tokens=1500" in model_args
    assert "min_visual_tokens=" not in model_args


def test_lmms_eval_model_args_default_quant_lambda_is_half(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.delenv("PQ_GAE_QUANT_LAMBDA", raising=False)

    args = Namespace(model_args="", model_path="")
    model_args = module._build_default_model_args(args)

    assert "gae_quant_lambda=0.5" in model_args


def test_lmms_eval_explicit_model_args_win() -> None:
    module = _load_runner_module()
    args = Namespace(model_args="pretrained=/custom,retention_ratio=0.25", model_path="/ignored")

    assert module._build_default_model_args(args) == "pretrained=/custom,retention_ratio=0.25"


def test_lmms_eval_env_adds_model_plugin_for_legacy_model_registry(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.delenv("LMMS_EVAL_PLUGINS", raising=False)

    env = module._build_subprocess_env("/repo", "/repo/third_party/lmms-eval")

    assert env["LMMS_EVAL_PLUGINS"] == "prune_quant_baseline.lmms_eval"


def test_lmms_eval_env_defaults_to_local_hf_dataset_cache(monkeypatch) -> None:
    module = _load_runner_module()
    for name in (
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_HUB_CACHE",
        "LMMS_EVAL_DATASETS_CACHE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)

    env = module._build_subprocess_env("/repo", "/repo/third_party/lmms-eval")

    assert env["HF_HOME"] == "/home/aistudio/data/datasets/387822/abcd/hf_home"
    assert env["HF_DATASETS_CACHE"] == "/home/aistudio/data/datasets/387822/abcd/hf_home/datasets"
    assert env["HF_HUB_CACHE"] == "/home/aistudio/data/datasets/387822/abcd/hf_home/hub"
    assert env["LMMS_EVAL_DATASETS_CACHE"] == "/home/aistudio/data/datasets/387822/abcd/hf_home/datasets"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_lmms_eval_env_preserves_explicit_hf_cache_overrides(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    monkeypatch.setenv("HF_DATASETS_CACHE", "/custom/datasets")
    monkeypatch.setenv("HF_HUB_CACHE", "/custom/hub")
    monkeypatch.setenv("LMMS_EVAL_DATASETS_CACHE", "/custom/lmms-datasets")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    env = module._build_subprocess_env("/repo", "/repo/third_party/lmms-eval")

    assert env["HF_HOME"] == "/custom/hf"
    assert env["HF_DATASETS_CACHE"] == "/custom/datasets"
    assert env["HF_HUB_CACHE"] == "/custom/hub"
    assert env["LMMS_EVAL_DATASETS_CACHE"] == "/custom/lmms-datasets"
    assert env["HF_DATASETS_OFFLINE"] == "0"
    assert env["HF_HUB_OFFLINE"] == "0"


def test_lmms_eval_env_prefers_lmms_eval_hf_home_over_ambient_hf_home(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setenv("LMMS_EVAL_HF_HOME", "/mounted/hf_home")
    monkeypatch.setenv("HF_HOME", "/wrong/hf")
    monkeypatch.setenv("HF_DATASETS_CACHE", "/wrong/datasets")
    monkeypatch.setenv("HF_HUB_CACHE", "/wrong/hub")
    monkeypatch.setenv("LMMS_EVAL_DATASETS_CACHE", "/wrong/lmms-datasets")

    env = module._build_subprocess_env("/repo", "/repo/third_party/lmms-eval")

    assert env["HF_HOME"] == "/mounted/hf_home"
    assert env["HF_DATASETS_CACHE"] == "/mounted/hf_home/datasets"
    assert env["HF_HUB_CACHE"] == "/mounted/hf_home/hub"
    assert env["LMMS_EVAL_DATASETS_CACHE"] == "/mounted/hf_home/datasets"


def test_lmms_eval_resolves_local_hf_snapshot_from_ref(tmp_path: Path) -> None:
    module = _load_runner_module()
    repo_cache = tmp_path / "hub" / "datasets--lmms-lab--MMMU"
    snapshot = repo_cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123\n", encoding="utf-8")

    assert module._resolve_local_hf_dataset_snapshot(tmp_path, "lmms-lab/MMMU") == snapshot


def test_lmms_eval_prepares_local_task_overlay_for_cached_hf_dataset(tmp_path: Path) -> None:
    module = _load_runner_module()
    hf_home = tmp_path / "hf_home"
    repo_cache = hf_home / "hub" / "datasets--lmms-lab--MMMU"
    snapshot = repo_cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123\n", encoding="utf-8")
    output_path = tmp_path / "out"
    lmms_eval_root = Path(__file__).resolve().parents[1] / "third_party" / "lmms-eval"

    tasks, include_path, missing = module._prepare_local_task_overlays(
        tasks=["mmmu_val", "ocrbench"],
        lmms_eval_root=lmms_eval_root,
        hf_home=hf_home,
        output_path=output_path,
    )

    assert tasks == ["pq_local_mmmu_val", "ocrbench"]
    assert include_path == output_path / "_local_lmms_tasks"
    assert missing == [("ocrbench", "echo840/OCRBench")]
    overlay = (include_path / "mmmu_val.yaml").read_text(encoding="utf-8")
    assert "dataset_path: parquet" in overlay
    assert f"include: {lmms_eval_root / 'lmms_eval' / 'tasks' / 'mmmu' / 'mmmu_val.yaml'}" in overlay
    assert "local_files_only: true" in overlay
    assert "dataset_kwargs:\n  data_files:" in overlay
    assert f"    validation: {snapshot / 'data' / 'validation-*'}" in overlay
    assert f"  cache_dir: {output_path / '_hf_datasets_cache'}" in overlay
    assert "task: pq_local_mmmu_val" in overlay


def test_lmms_eval_requires_local_snapshots_by_default(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.delenv("LMMS_EVAL_ALLOW_HUB_FALLBACK", raising=False)

    assert module._require_local_snapshots({"HF_HUB_OFFLINE": "1"})
    assert not module._require_local_snapshots({"HF_HUB_OFFLINE": "0"})


def test_lmms_eval_can_allow_hub_fallback(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setenv("LMMS_EVAL_ALLOW_HUB_FALLBACK", "1")

    assert not module._require_local_snapshots({"HF_HUB_OFFLINE": "1"})


def test_lmms_eval_model_plugin_exposes_empty_tasks_package() -> None:
    spec = importlib.util.find_spec("prune_quant_baseline.lmms_eval.tasks")

    assert spec is not None
    assert spec.submodule_search_locations is not None
