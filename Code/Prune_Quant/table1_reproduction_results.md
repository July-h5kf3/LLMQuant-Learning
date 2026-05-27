# Table 1 Reproduction Results

Date: 2026-05-27

Model: Qwen2-VL-7B-Instruct

Evaluation: VLMEvalKit

Image token setting:
- Approx-1500: `min_pixels=1003520`, `max_pixels=1176000`
- Fixed-1500: `min_pixels=max_pixels=1176000`

## Main Results

| Dataset | Metric | Paper 100% | Paper GAE 50% | Our 100% approx-1500 | Our GAE 50% approx-1500 | Notes |
|---|---:|---:|---:|---:|---:|---|
| MME | total | 2295.1 | 2297.1 | 2284.4 | 2292.6 | Official VLMEvalKit MME data verified earlier. |
| MME | perception | - | - | 1698.0 | 1698.6 |  |
| MME | reasoning | - | - | 586.4 | 593.9 |  |
| MMStar | overall | 60.4 | 60.3 | 56.6 | 56.8 | Exact matching. |
| MMVet | overall | 54.0 | 53.2 | 41.3 | 42.5 | DeepSeek judge, not GPT judge. Absolute values are not paper-comparable. |

## Additional Checks

| Check | Result | Interpretation |
|---|---:|---|
| MMStar official Qwen2VLChat vanilla, approx-1500 | 56.27 | Official wrapper is also around 56.x, so the MMStar gap is likely not caused by our GAE wrapper. |
| MMVet fixed-1500 100%, DeepSeek judge | 43.9 | Earlier fixed-resolution run. |
| MMVet fixed-1500 GAE 50%, DeepSeek judge | 44.1 | Earlier fixed-resolution run. |

## Source Files On Remote

| Run | Remote file |
|---|---|
| MME 100% approx-1500 | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mme_approx1500/Qwen2VL_GAE100_Approx1500/T20260527-154125/Qwen2VL_GAE100_Approx1500_MME_score.csv` |
| MME GAE 50% approx-1500 | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mme_approx1500_gae50_rerun/Qwen2VL_GAE50_Approx1500/T20260527-173027/Qwen2VL_GAE50_Approx1500_MME_score.csv` |
| MMStar 100% approx-1500 | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mmstar_approx1500_gae100/Qwen2VL_GAE100_Approx1500/T20260527-173027/Qwen2VL_GAE100_Approx1500_MMStar_acc.csv` |
| MMStar GAE 50% approx-1500 | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mmstar_approx1500_gae50_parallel/Qwen2VL_GAE50_Approx1500/T20260527-154908/Qwen2VL_GAE50_Approx1500_MMStar_acc.csv` |
| MMStar official vanilla sanity check | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mmstar_official_vanilla_approx1500_eager4/Qwen2VL_Official_Approx1500_Eager/T20260527-192632/Qwen2VL_Official_Approx1500_Eager_MMStar_acc.csv` |
| MMVet 100% approx-1500, DeepSeek judge | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mmvet_approx1500_gae100/Qwen2VL_GAE100_Approx1500/T20260527-185052/Qwen2VL_GAE100_Approx1500_MMVet_deepseek-v4-pro-mt4096_score.csv` |
| MMVet GAE 50% approx-1500, DeepSeek judge | `/root/autodl-tmp/vlmeval_runs/qwen2vl_mmvet_approx1500_gae50/Qwen2VL_GAE50_Approx1500/T20260527-190408/Qwen2VL_GAE50_Approx1500_MMVet_deepseek-v4-pro-mt4096_score.csv` |

## Notes

- MMStar official vanilla sanity check required changing the remote VLMEvalKit Qwen2VL wrapper from `flash_attention_2` to `eager`, because `flash_attn` is not installed on the machine. A backup exists remotely at `vlmeval/vlm/qwen2_vl/model.py.flashattn.bak`.
- MMVet was judged with `deepseek-v4-pro` using a custom scorer with larger `max_tokens`; parse failures were 0 for both approx-1500 runs.
- No remote `run.py` or DeepSeek judge process was left running after these results were collected.
