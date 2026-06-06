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

    assert module.DEFAULT_TASKS == ("ocrbench", "vizwiz_vqa_val", "scienceqa_img", "textvqa_val")


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


def test_lmms_eval_model_plugin_exposes_empty_tasks_package() -> None:
    spec = importlib.util.find_spec("prune_quant_baseline.lmms_eval.tasks")

    assert spec is not None
    assert spec.submodule_search_locations is not None
