# Gate 2.4f Structured Oracle Readout Evaluation

- Date: 2026-06-08
- Status: completed
- Device: NVIDIA GeForce RTX 5090
- Purpose: re-evaluate existing Gate 2.4d / Gate 2.4e sample scorer checkpoints
  with flat, `SE(3)`, and `SE(3)+gripper` oracle definitions.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |
| split policy | episode |
| K | 16 prior samples |

## Code Changes

- `scripts/train_visual_cvae_sample_scorer.py`
  - added flat alias metrics;
  - added `se3_oracle_best_*` metrics;
  - added `se3_gripper_oracle_best_*` metrics;
  - added scorer top-1 match, rank, and score-regret under structured oracle
    definitions.
- `scripts/evaluate_visual_cvae_sample_scorer.py`
  - added a checkpoint-only evaluation entry point, so existing scorer models
    can be re-evaluated without retraining.
- `tests/test_future_motion_predictor.py`
  - added coverage for structured oracle scores and rank direction.

## Commands

Template:

```bash
.venv/bin/python scripts/evaluate_visual_cvae_sample_scorer.py \
  --scorer-checkpoint <scorer-model.pt> \
  --output-json <gate2_4f-output.json> \
  --device cuda \
  --seed <seed> \
  --quiet
```

Concrete scorer checkpoints:

| branch | seed | scorer checkpoint | output |
| --- | ---: | --- | --- |
| flat | 7 | `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed7/model.pt` | `outputs/visual_cvae_sample_scorer_eval/gate2_4f_flat_seed7.json` |
| flat | 17 | `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed17/model.pt` | `outputs/visual_cvae_sample_scorer_eval/gate2_4f_flat_seed17.json` |
| `se3` | 7 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed7/model.pt` | `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_seed7.json` |
| `se3` | 17 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed17/model.pt` | `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_seed17.json` |
| `se3_gripper` | 7 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed7/model.pt` | `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_gripper_seed7.json` |
| `se3_gripper` | 17 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed17/model.pt` | `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_gripper_seed17.json` |

## Results

Mean over seed 7 and seed 17:

| scorer target | action MSE | trans L2 m | rot geod deg | gripper MSE | flat rank | flat top1 | SE(3) rank | SE(3) top1 | SE(3)+gripper rank | SE(3)+gripper top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat action MSE | 0.040190 | 0.014229 | 2.093158 | 0.165335 | 6.624765 | 0.235384 | 7.768547 | 0.181146 | 7.451381 | 0.180215 |
| `SE(3)` | 0.040642 | 0.014306 | 2.037711 | 0.167648 | 7.563506 | 0.173504 | 7.289575 | 0.193948 | 7.663167 | 0.161680 |
| `SE(3)+gripper` | 0.040441 | 0.014308 | 2.042545 | 0.166296 | 7.220221 | 0.183950 | 7.408181 | 0.184478 | 7.364062 | 0.177226 |

Oracle diagnostics, mean over the same sample sets:

| oracle selection | action MSE | trans L2 m | rot geod deg | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| flat oracle best-of-K | 0.036894 | - | - | - |
| `SE(3)` oracle best-of-K | 0.038628 | 0.013566 | 1.842173 | - |
| `SE(3)+gripper` oracle best-of-K | 0.037801 | - | - | 0.156192 |

Gap closure from prior mean action MSE `0.040931` to flat oracle best-of-K
action MSE `0.036894`:

| scorer target | gap closed |
| --- | ---: |
| flat action MSE | 18.36% |
| `SE(3)` | 7.16% |
| `SE(3)+gripper` | 12.15% |

## Interpretation

The structured scorer targets do improve the diagnostic they are trained toward:

- `SE(3)` scorer has the best `SE(3)` oracle rank: `7.289575`;
- `SE(3)+gripper` scorer has the best `SE(3)+gripper` oracle rank: `7.364062`.

However, neither structured target beats the flat scorer on deployable action
MSE. The `SE(3)` target improves rotation geodesic error, but pays for it in
flat action MSE and gripper MSE. The `SE(3)+gripper` target partially recovers
the structured rank but still does not beat the flat scorer.

This means Gate 2.4e did not only look weak because the old oracle/rank was
flat. It remains weaker after adding structured oracle diagnostics.

## Decision

Gate 2.4f confirms the mainline order:

1. keep Gate 2.4d flat action-MSE scorer as the current deployable readout
   baseline;
2. keep `SE(3)` / `SE(3)+gripper` oracle rank as diagnostics;
3. move next to Gate 2.4g hard-negative / executability-aware readout;
4. do not spend more mainline time on naive metric-target replacement.

## Verification

```text
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src scripts tests
.venv/bin/ruff check src scripts tests
```

All checks passed before GPU evaluation.
