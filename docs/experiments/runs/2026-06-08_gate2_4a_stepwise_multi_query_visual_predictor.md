# Gate 2.4a Stepwise Multi-Query Visual Predictor

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4a
- Purpose: test whether horizon-step-specific DINO patch attention improves the
  action-executable visual future-motion prior.

## Dataset Slice

Source:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Visual cache:

```text
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 task files | 8 |
| windows | 16,518 |
| context dim | 15 |
| future-motion dim | 48 |
| future horizon | 8 |
| future step dim | 6 |
| action dim | 7 |
| split policy | episode |

## Method

Previous visual cross-attention used one query for the whole future horizon:

```text
proprio + suite_task -> query
query attends DINO patch tokens -> one grounded token g_t
[proprio, suite_task, g_t] -> full future_delta_ee chunk
```

Gate 2.4a uses one query per future step:

```text
proprio + suite_task -> base query
base query + learned step embedding[k] -> query_k
query_k attends DINO patch tokens -> grounded token g_{t,k}
[proprio, suite_task, g_{t,k}, step embedding[k]] -> future_delta_ee[k]
```

Training objective:

```text
motion_loss = MSE(pred_future_ee_delta, gt_future_ee_delta)
action_loss = MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)
total_loss = motion_loss + 0.03 * action_loss
```

## Code Changes

- `src/geomoco_wm/models/future_motion_predictor.py`
  - added `StepwiseVisualCrossAttentionFutureMotionPredictor`;
  - it reshapes the 48D future-motion target into `8 x 6` step outputs;
  - it uses learned step embeddings to create one attention query per future
    step.
- `scripts/train_future_motion_predictor.py`
  - added `--visual-fusion stepwise_cross_attention`;
  - added `--future-step-dim`, default `6`.
- `tests/test_future_motion_predictor.py`
  - added shape and divisibility tests for the stepwise predictor.

## Model And Training Config

```text
script: scripts/train_future_motion_predictor.py
model: StepwiseVisualCrossAttentionFutureMotionPredictor
visual fusion: stepwise_cross_attention
visual tokens: 64 x 384D
horizon queries: 8
future step dim: 6
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
weight decay: 0
split policy: episode
seed(s): 7, 17
device: cuda
action-aware loss weight: 0.03
downstream decoder: frozen Gate 1.6 geodesic oracle future-motion ActionDecoder
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion stepwise_cross_attention \
  --output-dir outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt \
  --action-aware-loss-weight 0.03
```

Seed 17:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion stepwise_cross_attention \
  --output-dir outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt \
  --action-aware-loss-weight 0.03
```

## Artifacts

| seed | metrics | checkpoint |
| ---: | --- | --- |
| 7 | `outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed7/model.pt` |
| 17 | `outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed17/model.pt` |

## Per-Seed Results

| seed | future MSE | future trans L2 | future orient L2 | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.000774 | 0.017728 | 0.050858 | 0.043687 | 0.112240 | 0.014684 | 1.977688 | 0.185739 |
| 17 | 0.000777 | 0.016582 | 0.051305 | 0.041688 | 0.112360 | 0.014816 | 2.107074 | 0.170053 |

## Mean Results

| branch | future MSE | future trans L2 | future orient L2 | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.3a single-query lambda 0.010 | 0.000770 | 0.016763 | 0.049942 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 | 66.12% |
| Gate 2.3b single-query lambda 0.030 | 0.000782 | 0.018767 | 0.050640 | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |
| Gate 2.4a stepwise-query lambda 0.030 | 0.000776 | 0.017155 | 0.051082 | 0.042687 | 0.112300 | 0.014750 | 2.042381 | 0.177896 | 67.53% |

## Interpretation

Gate 2.4a is mixed.

The stepwise multi-query model improves future-motion translation geometry
relative to the single-query `lambda_action=0.030` branch:

```text
single-query lambda 0.030 future trans L2: 0.018767
stepwise-query lambda 0.030 future trans L2: 0.017155
```

But it does not improve the primary action-value metric:

```text
single-query lambda 0.030 action MSE: 0.042090
stepwise-query lambda 0.030 action MSE: 0.042687
```

So the result should not replace the Gate 2.3b default. The current best
action-value deterministic prior remains:

```text
single-query cross_attention + lambda_action=0.030
```

The stepwise branch is still useful as evidence: per-step visual querying helps
motion-space translation consistency, but this alone is not enough to improve
the frozen action decoder interface. The remaining oracle gap likely needs
multimodality, gripper/contact modeling, or an action-aware decoder with
stronger temporal structure.

## Decision

Do not promote Gate 2.4a as the default action-value branch.

Keep it as a geometry-balanced alternative and an architectural ingredient for
later cVAE / stochastic models, where per-step queries may help generate
diverse future trajectories.

## Next Decision

Proceed to the stochastic / multimodal prior branch. The most sensible next
step is a visual-conditioned cVAE future-motion prior with:

1. single-query `lambda_action=0.030` as the deterministic action-value
   baseline;
2. optional stepwise queries as a geometry-preserving variant;
3. frozen action-decoder auxiliary loss reported in addition to reconstruction
   and KL terms;
4. gripper/contact diagnostics kept separate from EEF SE(3) geometry.
