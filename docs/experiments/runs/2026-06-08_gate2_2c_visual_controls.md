# Gate 2.2c Visual Controls

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.2c
- Purpose: verify that Gate 2.2b's visual gain comes from aligned visual
  grounding, and identify camera-specific contribution before promoting the
  path into stronger priors.

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

## Visual Caches

| branch | cache | token config |
| --- | --- | --- |
| Shuffled two-camera | `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5` | 64 x 384 |
| Agentview-only | `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_agentview_2files_h8.h5` | 32 x 384 |
| Eye-in-hand-only | `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_eye_in_hand_2files_h8.h5` | 32 x 384 |
| Two-camera reference | `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5` | 64 x 384 |

The shuffled cache preserves the original `window_ids` order but assigns each
window another window's visual feature row. Its summary reports `fixed_points=0`.

## Model And Training Config

```text
script: scripts/train_future_motion_predictor.py
model: VisualCrossAttentionFutureMotionPredictor
visual fusion: cross_attention
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

## Commands

Create shuffled cache:

```bash
.venv/bin/python scripts/shuffle_visual_feature_cache.py \
  --input-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5 \
  --seed 7
```

Create camera-ablation caches:

```bash
.venv/bin/python scripts/cache_libero_dino_features.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_agentview_2files_h8.h5 \
  --feature-mode patch_pool \
  --patch-pool-grid 4 \
  --camera-keys agentview_rgb \
  --batch-size 128 \
  --device cuda
```

```bash
.venv/bin/python scripts/cache_libero_dino_features.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_eye_in_hand_2files_h8.h5 \
  --feature-mode patch_pool \
  --patch-pool-grid 4 \
  --camera-keys eye_in_hand_rgb \
  --batch-size 128 \
  --device cuda
```

Training command template:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache <CACHE> \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/<RUN_TAG> \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed <SEED> \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed<SEED>/model.pt
```

## Artifacts

| branch | seed | metrics | checkpoint |
| --- | ---: | --- | --- |
| Shuffled two-camera | 7 | `outputs/future_motion_predictor/gate2_2c_shuffled_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_2c_shuffled_patchpool4_crossattn_seed7/model.pt` |
| Shuffled two-camera | 17 | `outputs/future_motion_predictor/gate2_2c_shuffled_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_2c_shuffled_patchpool4_crossattn_seed17/model.pt` |
| Agentview-only | 7 | `outputs/future_motion_predictor/gate2_2c_agentview_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_2c_agentview_patchpool4_crossattn_seed7/model.pt` |
| Agentview-only | 17 | `outputs/future_motion_predictor/gate2_2c_agentview_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_2c_agentview_patchpool4_crossattn_seed17/model.pt` |
| Eye-in-hand-only | 7 | `outputs/future_motion_predictor/gate2_2c_eye_in_hand_patchpool4_crossattn_seed7/metrics.json` | `outputs/future_motion_predictor/gate2_2c_eye_in_hand_patchpool4_crossattn_seed7/model.pt` |
| Eye-in-hand-only | 17 | `outputs/future_motion_predictor/gate2_2c_eye_in_hand_patchpool4_crossattn_seed17/metrics.json` | `outputs/future_motion_predictor/gate2_2c_eye_in_hand_patchpool4_crossattn_seed17/model.pt` |

## Per-Seed Results

| Branch | Seed | Future MSE | Action MSE | Action MAE | Translation L2 | Rotation geodesic | Gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shuffled two-camera | 7 | 0.000913 | 0.072458 | 0.149156 | 0.019159 m | 2.036370 deg | 0.306423 |
| Shuffled two-camera | 17 | 0.001057 | 0.078585 | 0.160843 | 0.021997 m | 2.219114 deg | 0.294531 |
| Agentview-only | 7 | 0.000817 | 0.058404 | 0.131550 | 0.016511 m | 1.965750 deg | 0.253463 |
| Agentview-only | 17 | 0.000792 | 0.051060 | 0.124220 | 0.016161 m | 2.062424 deg | 0.212809 |
| Eye-in-hand-only | 7 | 0.000790 | 0.054097 | 0.127806 | 0.015446 m | 1.973731 deg | 0.249017 |
| Eye-in-hand-only | 17 | 0.000808 | 0.047608 | 0.120432 | 0.015420 m | 2.075513 deg | 0.200833 |

## Mean Results

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| Shuffled two-camera patch | 0.000985 | 0.075521 | 0.154999 | 0.020578 | 2.127742 | 0.300477 |
| Agentview-only patch | 0.000804 | 0.054732 | 0.127885 | 0.016336 | 2.014087 | 0.233136 |
| Eye-in-hand-only patch | 0.000799 | 0.050853 | 0.124119 | 0.015433 | 2.024622 | 0.224925 |
| Two-camera patch reference | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| Direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| Oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Interpretation

The shuffled control fails in the expected direction:

```text
shuffled action MSE: 0.075521
aligned two-camera action MSE: 0.049547
```

This means the visual branch is not winning merely because the model sees a
large DINO feature vector or because the cache contains generic dataset
statistics. Correct window-to-visual alignment matters.

Both real single-camera branches beat direct context:

```text
agentview-only: 0.054732 < 0.066010
eye-in-hand-only: 0.050853 < 0.066010
```

Eye-in-hand is slightly stronger than agentview on this slice, while the
two-camera cache is still best overall. The camera result suggests that the
local wrist view carries useful manipulation geometry, but combining it with
agentview still gives a small extra gain.

## Limits

- The shuffled cache uses one derangement seed. It is strong enough as a first
  negative control, but a later paper table can add more shuffle seeds.
- Camera ablations use the same model size. The two-camera branch has twice as
  many visual tokens as a single-camera branch.
- This still evaluates predictive action value through a frozen MLP
  action-decoder interface, not closed-loop LIBERO success.

## Next Decision

Gate 2.2 controls pass. Use the aligned two-camera patch cross-attention branch
as the default visual grounding route. The next mainline step should reduce the
remaining oracle gap with an action-aware or multimodal future-motion prior
before attaching the path to GeoMoCo-cVAE.
