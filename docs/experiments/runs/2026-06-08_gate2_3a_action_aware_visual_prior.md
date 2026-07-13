# Gate 2.3a Action-Aware Visual Future-Motion Prior

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.3a
- Purpose: test whether a frozen action-decoder auxiliary loss makes the
  aligned visual future-motion prior more useful to the downstream action
  interface.

## Dataset Slice

Source:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 task files | 8 |
| windows | 16,518 |
| context dim | 15 |
| future-motion dim | 48 |
| action dim | 7 |
| horizon | 8 |
| split policy | episode |

## Method

Gate 2.2b trained only with future-motion MSE:

```text
motion_loss = MSE(pred_future_ee_delta, gt_future_ee_delta)
```

Gate 2.3a adds a frozen action-decoder auxiliary loss:

```text
action_loss = MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)
total_loss = motion_loss + 0.01 * action_loss
```

The frozen action decoder is the same Gate 1.6 geodesic oracle future-motion
decoder used for downstream evaluation. Its parameters are frozen; gradients
flow through it into the future-motion predictor.

## Model And Training Config

```text
script: scripts/train_future_motion_predictor.py
model: VisualCrossAttentionFutureMotionPredictor
visual cache: outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
visual fusion: cross_attention
visual tokens: 64 x 384D
query: MLP(proprio + suite_task one-hot) -> 384D
attention heads: 4
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
weight decay: 0
split policy: episode
seed(s): 7, 17
device: cuda
action-aware loss weight: 0.01
downstream decoder: frozen Gate 1.6 geodesic oracle future-motion ActionDecoder
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt \
  --action-aware-loss-weight 0.01
```

Seed 17:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt \
  --action-aware-loss-weight 0.01
```

## Artifacts

| seed | metrics | checkpoint |
| ---: | --- | --- |
| 7 | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed7/model.pt` |
| 17 | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed17/model.pt` |

## Results

Per-seed downstream action metrics:

| branch | seed | future MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.2b visual MSE-only | 7 | 0.000748 | 0.052329 | 0.123296 | 0.014743 | 1.944853 | 0.244798 |
| Gate 2.2b visual MSE-only | 17 | 0.000797 | 0.046764 | 0.117443 | 0.014976 | 2.116048 | 0.200136 |
| Gate 2.3a action-aware | 7 | 0.000786 | 0.043855 | 0.112611 | 0.014413 | 1.997400 | 0.188922 |
| Gate 2.3a action-aware | 17 | 0.000755 | 0.042493 | 0.114252 | 0.015256 | 2.077536 | 0.166939 |

Mean metrics:

| branch | future MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.2b visual MSE-only | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| Gate 2.3a action-aware | 0.000770 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 |
| Direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| Oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

Relative readout:

```text
action MSE improvement over Gate 2.2b: 12.86%
direct-to-oracle action-MSE gap closure:
  Gate 2.2b: 47.67%
  Gate 2.3a: 66.12%
```

## Interpretation

Gate 2.3a is positive. A small action-aware loss improves downstream action MSE
substantially while keeping future-motion MSE almost unchanged.

The largest surprise is gripper MSE:

```text
Gate 2.2b gripper MSE: 0.222467
Gate 2.3a gripper MSE: 0.177930
Oracle future-motion gripper MSE: 0.184683
```

This does not mean the learned prior beats the oracle in a strict sense; the
oracle row uses a decoder trained only to map GT future EEF motion to action,
while Gate 2.3a directly optimizes the predicted motion for that frozen decoder.
It does mean the action-aware objective can reshape predicted motion into a
more executable intermediate variable, especially for gripper-relevant action
readout.

## Limits

- Only `lambda_action=0.01` was tested.
- This is still a deterministic future-motion prior, not a multimodal prior.
- The auxiliary action loss uses the same frozen MLP action decoder as the
  evaluation interface; this is an attribution-clean mechanism test, not a
  final policy architecture.
- Translation future-motion L2 is slightly worse than Gate 2.2b even though
  action translation L2 is nearly unchanged. This is acceptable for this gate,
  but future variants should report both motion-space and action-space metrics.

## Next Decision

Promote action-aware loss as the next mainline branch. Next experiments:

1. Sweep `lambda_action` around `0.003`, `0.01`, and `0.03`.
2. Add step-wise / multi-query visual attention.
3. Add a gripper/contact auxiliary branch.
4. Then evaluate stochastic or multimodal future-motion priors.
