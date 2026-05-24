import pytest

from prune_quant_baseline.core.config import load_config


def test_config_loading_env_and_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/remote/model")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  model_type: qwen2vl
  model_path: ${MODEL_PATH}
data:
  input_jsonl: ${INPUT_JSONL}
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["model"]["model_path"] == "/remote/model"
    assert cfg["data"]["input_jsonl"] == "${INPUT_JSONL}"
    assert cfg["quant"]["method"] == "none"
    assert cfg["pruning"]["retention_ratio"] == 0.5


def test_config_strict_env_errors(tmp_path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  model_type: qwen2vl
  model_path: ${MISSING_MODEL_PATH}
""",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="MISSING_MODEL_PATH"):
        load_config(cfg_path, strict_env=True)


def test_config_missing_required_field_errors(tmp_path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  model_type: qwen2vl
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model_path"):
        load_config(cfg_path)
