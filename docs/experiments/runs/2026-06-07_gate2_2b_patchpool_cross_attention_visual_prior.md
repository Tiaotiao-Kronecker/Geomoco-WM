# Gate 2.2b Patch-Pooled DINOv2 Cross-Attention Visual Prior

- Date: 2026-06-07
- Status: completed
- Gate: Gate 2.2b
- Purpose: test whether patch-level DINOv2 visual tokens with cross-attention
  improve the visual-grounded future-motion prior beyond Gate 2.2a global
  tokens.

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

## Visual Cache

Cache:

```text
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.summary.json
```

Config:

| field | value |
| --- | --- |
| model | `dinov2_vits14_reg` |
| source | local torchhub cache |
| feature mode | `patch_pool_4x4_context_camera_concat` |
| cameras | `agentview_rgb`, `eye_in_hand_rgb` |
| context frames | 2 |
| pooled patch tokens per image | 16 |
| visual token count per window | 64 |
| visual token dim | 384 |
| flat cache dim | 24,576 |
| image size | 224 |
| cache device | cuda |

The cache spatially pools the 16x16 DINO patch grid into a 4x4 grid per image.
This keeps patch-level grounding affordable while avoiding a raw full-patch
cache of roughly 25GB for this slice.

## Model And Training Config

```text
script: scripts/train_future_motion_predictor.py
model: VisualCrossAttentionFutureMotionPredictor
visual fusion: cross_attention
query: MLP(proprio + suite_task one-hot) -> 384D
visual tokens: 64 x 384D
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

## Commands

Smoke cache:

```bash
.venv/bin/python scripts/cache_libero_dino_features.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-path outputs/visual_features/gate2_2b_dinov2_patchpool_smoke.h5 \
  --feature-mode patch_pool \
  --patch-pool-grid 4 \
  --max-windows 8 \
  --batch-size 4 \
  --device cuda
```

Full cache:

```bash
.venv/bin/python scripts/cache_libero_dino_features.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --feature-mode patch_pool \
  --patch-pool-grid 4 \
  --batch-size 128 \
  --device cuda
```

Seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/gate2_2b_patchpool4_crossattn_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Seed 17:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/gate2_2b_patchpool4_crossattn_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Results

Future-motion validation metrics:

| seed | val MSE | val MAE | val L2 | trans L2 | orient coord L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.000748 | 0.015477 | 0.165176 | 0.014504 | 0.049178 |
| 17 | 0.000797 | 0.015943 | 0.170882 | 0.014378 | 0.051050 |
| mean | 0.000772 | 0.015710 | 0.168029 | 0.014441 | 0.050114 |

Downstream frozen action-decoder metrics from predicted motion:

| seed | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.052329 | 0.123296 | 0.014743 | 1.944853 | 0.244798 |
| 17 | 0.046764 | 0.117443 | 0.014976 | 2.116048 | 0.200136 |
| mean | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |

## Comparison To Gate 2.2a

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.2a DINO global visual prior | 0.000801 | 0.053628 | 0.128207 | 0.016285 | 2.031589 | 0.229947 |
| Gate 2.2b patch cross-attention prior | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

Relative readout:

- future-motion MSE improves by `3.62%` over Gate 2.2a.
- downstream action MSE improves by `7.61%` over Gate 2.2a.
- downstream action MSE improves by `24.94%` over direct context.
- Gate 2.2b closes `47.67%` of the direct-context to oracle-future-motion MSE
  gap.

## Interpretation

Patch-pooled cross-attention improves over global-token visual conditioning,
but the gain is moderate rather than dramatic. This suggests that spatial
grounding helps, while the current 4x4 pooling and simple single-query
attention may still be a bottleneck.

Gate 2.2b strengthens the visual-grounding thesis:

```text
patch visual grounding > global visual grounding > task/proprio only
```

under the frozen action-decoder interface.

## Limits

- This is pooled patch grounding, not full 16x16 patch attention.
- It uses one query per window; it does not yet model different future motion
  steps with separate visual queries.
- No shuffled-vision or camera ablation controls have been run yet.
- This is still deterministic, not cVAE.

## Next Decision

Before cVAE, run visual controls:

```text
shuffled patch/global visual features
agentview-only vs eye-in-hand-only vs two-camera
optional step-wise query attention
```

Then promote the best visual grounding path into GeoMoCo-cVAE.

