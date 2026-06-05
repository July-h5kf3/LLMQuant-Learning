from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_quant_joint_runner_defaults_lambda_to_half() -> None:
    for rel_path in (
        "remote/run_qwen2vl_quant_joint_rtn_mme.sh",
        "remote/run_masquant_pseudo_pipeline.sh",
    ):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")

        assert 'GAE_QUANT_LAMBDA="${GAE_QUANT_LAMBDA:-0.5}"' in text
