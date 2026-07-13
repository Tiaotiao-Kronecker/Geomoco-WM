# Gate 2.3b Action-Aware Lambda Sweep

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.3b
- Purpose: select the action-aware auxiliary loss weight for the aligned
  two-camera DINO patch cross-attention future-motion prior.

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

Gate 2.3b keeps the same visual branch as Gate 2.2b and the same frozen
oracle future-motion action decoder as Gate 2.3a:

```text
motion_loss = MSE(pred_future_ee_delta, gt_future_ee_delta)
action_loss = MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)
total_loss = motion_loss + lambda_action * action_loss
```

The sweep tests:

| lambda_action | reason |
| ---: | --- |
| 0.003 | weaker action signal, expected to preserve motion geometry better |
| 0.010 | Gate 2.3a reference |
| 0.030 | stronger action signal, expected to improve action value but may distort motion geometry |

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
downstream decoder: frozen Gate 1.6 geodesic oracle future-motion ActionDecoder
```

## Command Template

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/<run_tag> \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed <seed> \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed<seed>/model.pt \
  --action-aware-loss-weight <lambda_action>
```

## Artifacts

| lambda | seed | metrics | checkpoint |
| ---: | ---: | --- | --- |
| 0.003 | 7 | `outputs/future_motion_predictor/gate2_3b_action_aware_lam0003_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_3b_action_aware_lam0003_patchpool4_crossattn_seed7/model.pt` |
| 0.003 | 17 | `outputs/future_motion_predictor/gate2_3b_action_aware_lam0003_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_3b_action_aware_lam0003_patchpool4_crossattn_seed17/model.pt` |
| 0.010 | 7 | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed7/model.pt` |
| 0.010 | 17 | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed17/model.pt` |
| 0.030 | 7 | `outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed7/model.pt` |
| 0.030 | 17 | `outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed17/model.pt` |

## Per-Seed Results

| branch | seed | future MSE | future trans L2 | future orient L2 | action MSE | action MAE | action trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda 0.000, Gate 2.2b | 7 | 0.000748 | 0.014504 | 0.049178 | 0.052329 | 0.123296 | 0.014743 | 1.944853 | 0.244798 |
| lambda 0.000, Gate 2.2b | 17 | 0.000797 | 0.014378 | 0.051050 | 0.046764 | 0.117443 | 0.014976 | 2.116048 | 0.200136 |
| lambda 0.003 | 7 | 0.000795 | 0.015489 | 0.050763 | 0.047973 | 0.120669 | 0.015318 | 2.004591 | 0.206653 |
| lambda 0.003 | 17 | 0.000798 | 0.015505 | 0.050171 | 0.043608 | 0.116248 | 0.015340 | 2.143988 | 0.172571 |
| lambda 0.010 | 7 | 0.000786 | 0.016949 | 0.050649 | 0.043855 | 0.112611 | 0.014413 | 1.997400 | 0.188922 |
| lambda 0.010 | 17 | 0.000755 | 0.016576 | 0.049235 | 0.042493 | 0.114252 | 0.015256 | 2.077536 | 0.166939 |
| lambda 0.030 | 7 | 0.000792 | 0.019348 | 0.050953 | 0.041843 | 0.109251 | 0.014046 | 1.970044 | 0.182380 |
| lambda 0.030 | 17 | 0.000771 | 0.018186 | 0.050327 | 0.042338 | 0.112647 | 0.015150 | 2.063817 | 0.166659 |

## Mean Results

Reference:

```text
direct-context action MSE: 0.066010
oracle-future-motion action MSE: 0.031474
```

| branch | future MSE | future trans L2 | future orient L2 | action MSE | action MAE | action trans L2 (m) | rot geo (deg) | gripper MSE | direct-to-oracle gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct context | n/a | n/a | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 | 0.00% |
| Oracle future motion | n/a | n/a | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 | 100.00% |
| lambda 0.000, Gate 2.2b | 0.000772 | 0.014441 | 0.050114 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 | 47.67% |
| lambda 0.003 | 0.000796 | 0.015497 | 0.050467 | 0.045790 | 0.118458 | 0.015329 | 2.074289 | 0.189612 | 58.55% |
| lambda 0.010 | 0.000770 | 0.016763 | 0.049942 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 | 66.12% |
| lambda 0.030 | 0.000782 | 0.018767 | 0.050640 | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |

## Interpretation

The sweep is positive: all nonzero action-aware weights improve action MSE over
the MSE-only visual prior.

`lambda_action=0.030` gives the best downstream action value:

```text
action MSE: 0.042090
action MAE: 0.110949
action translation L2: 0.014598m
rotation geodesic: 2.016930deg
gripper MSE: 0.174519
direct-to-oracle gap closure: 69.26%
```

The cost is that predicted future-motion translation L2 worsens:

```text
Gate 2.2b lambda 0.000 future trans L2: 0.014441
lambda 0.010 future trans L2: 0.016763
lambda 0.030 future trans L2: 0.018767
```

This is not a collapse because future MSE and orientation-coordinate L2 remain
close, and all downstream action-space metrics improve. It does mean the
stronger action-aware objective is shaping the predicted future motion toward
the frozen decoder's executable manifold rather than purely minimizing
coordinate-space future-motion error.

## Decision

Use `lambda_action=0.030` as the action-value-prioritized deterministic
baseline for the next mainline branch.

Keep `lambda_action=0.010` as the balanced geometry reference. If a later
cVAE/multimodal run improves only action MSE while sharply degrading
future-motion geometry, compare against both `0.030` and `0.010`.

## Next Decision

Proceed to the multimodal prior branch with a validated visual-action route.
The next implementation should keep these controls:

1. Report both future-motion metrics and downstream action metrics.
2. Keep `lambda_action=0.030` as primary and `0.010` as a reference.
3. Add step-wise / multi-query visual conditioning or a stochastic prior before
   claiming that the remaining oracle gap is reducible.
4. Add gripper/contact auxiliary diagnostics because gripper action remains a
   separate bottleneck from SE(3) EEF geometry.
