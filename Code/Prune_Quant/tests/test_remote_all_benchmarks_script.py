from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "remote" / "run_qwen2vl_quant_joint_rtn_all_benchmarks.sh"
LMMS_TASKS = "mmmu_val ocrbench vizwiz_vqa_val scienceqa_img textvqa_val"


PURE_GAE_SCRIPT = REPO_ROOT / "remote" / "run_qwen2vl_pruned_gae_mme.example.sh"
LMMS_ONLY_SCRIPT = REPO_ROOT / "remote" / "run_qwen2vl_lmms_eval.example.sh"
MASQUANT_SCRIPT = REPO_ROOT / "remote" / "run_qwen2vl_masquant_pseudo_mme.example.sh"
GAE_MASQUANT_SCRIPT = REPO_ROOT / "remote" / "run_qwen2vl_masquant_gae50_pseudo_mme.sh"
PIPELINE_SCRIPT = REPO_ROOT / "remote" / "run_masquant_pseudo_pipeline.sh"
VANILLA_SCRIPT = REPO_ROOT / "remote" / "run_qwen2vl_vanilla_mme_mmstar.example.sh"


def test_all_benchmarks_script_covers_requested_benchmarks() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'export VLMEVAL_DATASETS="${VLMEVAL_DATASETS:-MME}"' in text
    assert f'export LMMS_EVAL_TASKS="${{LMMS_EVAL_TASKS:-{LMMS_TASKS}}}"' in text
    assert "remote/run_masquant_pseudo_pipeline.sh" in text
    assert "remote/run_lmms_eval_smart.py" in text


def test_all_benchmarks_script_caps_only_large_images_to_1500_visual_tokens() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'export PROCESSOR_MAX_VISUAL_TOKENS="${PROCESSOR_MAX_VISUAL_TOKENS:-1500}"' in text
    assert 'export PROCESSOR_MIN_VISUAL_TOKENS="${PROCESSOR_MIN_VISUAL_TOKENS:-}"' in text
    assert 'export PQ_MAX_VISUAL_TOKENS="$PROCESSOR_MAX_VISUAL_TOKENS"' in text
    assert 'export PQ_MIN_VISUAL_TOKENS="$PROCESSOR_MIN_VISUAL_TOKENS"' in text


def test_all_benchmarks_script_refinds_masquant_resume_after_child_pipeline() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ -z "${MASQUANT_RESUME:-}" ]]; then' in text
    assert 'find "$WORK_DIR/masquant_outputs" -name mas_parameters.pth' in text
    assert 'export MASQUANT_RESUME="$found_resume"' in text


def test_requested_configs_use_all_requested_lmms_eval_tasks() -> None:
    for script in (
        SCRIPT,
        PURE_GAE_SCRIPT,
        LMMS_ONLY_SCRIPT,
        MASQUANT_SCRIPT,
        GAE_MASQUANT_SCRIPT,
    ):
        text = script.read_text(encoding="utf-8")

        assert LMMS_TASKS in text
        if script in (MASQUANT_SCRIPT, GAE_MASQUANT_SCRIPT):
            assert 'export RUN_LMMS_EVAL="${RUN_LMMS_EVAL:-1}"' in text
        else:
            assert "remote/run_lmms_eval_smart.py" in text


def test_requested_vlmeval_configs_default_to_mme_only() -> None:
    for script in (SCRIPT, PURE_GAE_SCRIPT, MASQUANT_SCRIPT, GAE_MASQUANT_SCRIPT):
        text = script.read_text(encoding="utf-8")

        assert 'export VLMEVAL_DATASETS="${VLMEVAL_DATASETS:-MME}"' in text
        assert 'export VLMEVAL_EXACT_MATCH_DATASETS="${VLMEVAL_EXACT_MATCH_DATASETS:-MME}"' in text


def test_requested_configs_cap_large_images_without_forcing_small_images() -> None:
    for script in (SCRIPT, MASQUANT_SCRIPT, GAE_MASQUANT_SCRIPT):
        text = script.read_text(encoding="utf-8")

        assert 'export PROCESSOR_MIN_VISUAL_TOKENS="${PROCESSOR_MIN_VISUAL_TOKENS:-}"' in text
        assert 'export PROCESSOR_MAX_VISUAL_TOKENS="${PROCESSOR_MAX_VISUAL_TOKENS:-1500}"' in text

    for script in (PURE_GAE_SCRIPT, LMMS_ONLY_SCRIPT):
        text = script.read_text(encoding="utf-8")

        assert 'export PQ_MIN_VISUAL_TOKENS="${PQ_MIN_VISUAL_TOKENS:-}"' in text
        assert 'export PQ_MAX_VISUAL_TOKENS="${PQ_MAX_VISUAL_TOKENS:-1500}"' in text


def test_shared_masquant_pipeline_can_run_lmms_eval_when_enabled() -> None:
    text = PIPELINE_SCRIPT.read_text(encoding="utf-8")

    assert 'RUN_LMMS_EVAL="${RUN_LMMS_EVAL:-0}"' in text
    assert 'remote/run_lmms_eval_smart.py' in text
    assert 'export PQ_QUANT_METHOD=masquant' in text
    assert 'VLMEVAL_EXACT_MATCH_DATASETS="${VLMEVAL_EXACT_MATCH_DATASETS:-MME}"' in text


def test_full_and_pure_masquant_configs_default_to_accelerated_attention() -> None:
    vanilla_text = VANILLA_SCRIPT.read_text(encoding="utf-8")
    masquant_text = MASQUANT_SCRIPT.read_text(encoding="utf-8")
    pipeline_text = PIPELINE_SCRIPT.read_text(encoding="utf-8")

    assert 'export PQ_ATTN_IMPLEMENTATION="${PQ_ATTN_IMPLEMENTATION:-sdpa}"' in vanilla_text
    assert 'export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"' in masquant_text
    assert 'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"' in pipeline_text


def test_gae_configs_keep_eager_attention_for_attention_scores() -> None:
    for script in (PURE_GAE_SCRIPT, GAE_MASQUANT_SCRIPT, SCRIPT):
        text = script.read_text(encoding="utf-8")

        assert "eager" in text
        assert ":-sdpa" not in text
