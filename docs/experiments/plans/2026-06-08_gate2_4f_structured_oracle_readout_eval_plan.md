# Gate 2.4f Structured Oracle Readout Evaluation Plan

- Date: 2026-06-08
- Status: completed
- Position in mainline: after Gate 2.4e structured scorer, before Gate 2.4g
  hard-negative / executability-aware readout.

## Purpose

Gate 2.4e changed the scorer training target to `SE(3)` and `SE(3)+gripper`,
but the main oracle best-of-K and oracle rank diagnostics were still selected
with flat action MSE. This can hide whether a structured scorer is actually
better under a structured oracle.

Gate 2.4f is therefore an evaluation-only gate:

1. keep the trained Gate 2.4d / Gate 2.4e checkpoints fixed;
2. add `SE(3)` oracle best-of-K;
3. add `SE(3)+gripper` oracle best-of-K;
4. report scorer match/rank/regret under flat, `SE(3)`, and
   `SE(3)+gripper` oracle definitions.

## Why This Comes Before Gate 2.4g

The next mainline branch is hard-negative / executability-aware readout. Before
adding new supervision, we need to know which oracle definition should judge
sample readout quality.

If Gate 2.4e only looked weak because it was judged by flat MSE, then the next
branch should promote structured scorer targets. If it is still weak under the
structured oracle, then the bottleneck is not metric replacement; it is missing
sample-difficulty or executability information.

## Oracle Definitions

Flat oracle:

```text
argmin_k MSE(action_k, action_gt)
```

`SE(3)` oracle:

```text
argmax_k zscore(-translation_m_l2_k)
       + zscore(-rotation_geodesic_rad_k)
```

`SE(3)+gripper` oracle:

```text
argmax_k zscore(-translation_m_l2_k)
       + zscore(-rotation_geodesic_rad_k)
       + zscore(-gripper_mse_k)
```

The structured scores are standardized per batch/window candidate set, matching
the Gate 2.4e scorer target construction.

## Pass / Stop Criteria

Promote structured scorer target only if it improves at least one of:

- deployable action MSE;
- deployable `SE(3)` rank or top-1 match without severe action MSE regression;
- deployable gripper-aware rank or top-1 match without severe action MSE
  regression.

If structured scorer targets only improve diagnostic rank while action-value
metrics regress, keep the flat scorer as default and move to hard-negative or
executability-aware readout.

## Artifacts

Code:

```text
scripts/train_visual_cvae_sample_scorer.py
scripts/evaluate_visual_cvae_sample_scorer.py
tests/test_future_motion_predictor.py
```

Evaluation outputs:

```text
outputs/visual_cvae_sample_scorer_eval/gate2_4f_flat_seed7.json
outputs/visual_cvae_sample_scorer_eval/gate2_4f_flat_seed17.json
outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_seed7.json
outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_seed17.json
outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_gripper_seed7.json
outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_gripper_seed17.json
```
