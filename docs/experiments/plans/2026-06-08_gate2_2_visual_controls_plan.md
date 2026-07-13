# Gate 2.2 Visual Controls Plan

- Date: 2026-06-08
- Status: planned, then execute immediately
- Slice: four LIBERO suites, 2 HDF5 task files per suite, all demos, horizon 8
- Windows: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl`
- Split: episode-level train/validation split
- Seeds: 7 and 17

## Purpose

Gate 2.2b showed that patch-pooled DINOv2 visual tokens with cross-attention
improve the learned future-motion prior over direct context. Before connecting
this path to GeoMoCo-cVAE, this control gate checks whether the gain is truly
from visual grounding rather than feature leakage, dataset correlation, or a
single camera artifact.

## Reference Results

These are the current decision anchors:

| Branch | Mean action MSE | Mean action MAE | Translation L2 | Rotation geodesic | Gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct context lower bound | 0.066010 | 0.147124 | 0.019024 m | 2.233651 deg | 0.252545 |
| Gate 2.1 suite/task context prior | 0.072501 | 0.155573 | 0.020841 m | 2.106491 deg | 0.279382 |
| Gate 2.2a DINO global prior | 0.053628 | 0.128207 | 0.016285 m | 2.031589 deg | 0.229947 |
| Gate 2.2b DINO patch cross-attn prior | 0.049547 | 0.120370 | 0.014859 m | 2.030450 deg | 0.222467 |
| Oracle future motion upper bound | 0.031474 | 0.079508 | 0.007466 m | 1.048033 deg | 0.184707 |

## Control Matrix

Run all controls with the same future-motion predictor and frozen oracle action
decoder interface as Gate 2.2b:

| Control | Visual cache | Expected reading |
| --- | --- | --- |
| Two-camera patch baseline | existing patch-pooled cache | Reproduce Gate 2.2b reference. |
| Shuffled two-camera patch | same cache, feature rows deranged while window ids stay fixed | Should lose most of the visual gain. If it stays close to Gate 2.2b, suspect leakage or dataset shortcut. |
| Agentview-only patch | patch cache built only from `agentview_rgb` | Measures static scene / object-view contribution. |
| Eye-in-hand-only patch | patch cache built only from `eye_in_hand_rgb` | Measures wrist-view / manipulation-local contribution. |

## Commands

Create shuffled cache:

```bash
.venv/bin/python scripts/shuffle_visual_feature_cache.py \
  --input-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-path outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5 \
  --seed 7
```

Create agentview-only and eye-in-hand-only caches:

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

Train each control:

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

## Pass / Stop Criteria

Proceed beyond visual controls only if:

- shuffled features are clearly worse than the aligned two-camera patch cache;
- at least one real camera branch remains better than direct context or
  explains most of the two-camera gain;
- downstream action metrics, not only future-motion MSE, support the conclusion.

If shuffled features stay close to aligned visual features, stop and audit
window/cache alignment, train/val splits, and shortcut correlations before any
cVAE claim.

## Promotion Decision

If this gate passes, the next mainline step is not yet cVAE by default. First
use the control results to choose a reliable visual branch, then add an
action-aware or multimodal future-motion prior to reduce the remaining
Gate-2.2b-to-oracle gap. GeoMoCo-cVAE should receive the visual grounding path
after this attribution is clean.
