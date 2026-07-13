# Gate 2.2a DINOv2 Global Visual Future-Motion Prior

- Date: 2026-06-07
- Status: completed
- Gate: Gate 2.2a
- Purpose: test whether frozen DINOv2 global visual features make the learned
  future-motion prior useful as a downstream action-decoder interface.

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
outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5
outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.summary.json
```

Config:

| field | value |
| --- | --- |
| model | `dinov2_vits14_reg` |
| source | local torchhub cache |
| checkpoint | `/home/user/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth` |
| feature mode | `global_context_camera_concat` |
| cameras | `agentview_rgb`, `eye_in_hand_rgb` |
| context frames | 2 |
| DINO token dim | 384 |
| visual feature dim | 1,536 |
| image size | 224 |
| cache device | cuda |

Feature layout:

```text
[agentview_t0, eye_in_hand_t0, agentview_t1, eye_in_hand_t1]
```

Each element is a 384D DINOv2 global token. The concatenated vector is a
1536D visual feature.

## Model And Training Config

```text
script: scripts/train_future_motion_predictor.py
model: FutureMotionPredictor(context + suite_task one-hot + DINO global visual -> future_delta)
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

Conditioning:

| component | dim |
| --- | ---: |
| suite_task one-hot | 8 |
| DINO global visual | 1,536 |
| total conditioning | 1,544 |

## Commands

Visual cache smoke:

```bash
.venv/bin/python scripts/cache_libero_dino_features.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-path outputs/visual_features/gate2_2a_dinov2_smoke.h5 \
  --max-windows 8 \
  --batch-size 4 \
  --device cuda
```

Full visual cache:

```bash
.venv/bin/python scripts/cache_libero_dino_features.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-path outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5 \
  --batch-size 128 \
  --device cuda
```

Seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5 \
  --output-dir outputs/future_motion_predictor/gate2_2a_dinov2_global_seed7 \
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
  --visual-feature-cache outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5 \
  --output-dir outputs/future_motion_predictor/gate2_2a_dinov2_global_seed17 \
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
| 7 | 0.000824 | 0.016500 | 0.174012 | 0.016841 | 0.050999 |
| 17 | 0.000779 | 0.015856 | 0.168331 | 0.015597 | 0.049620 |
| mean | 0.000801 | 0.016178 | 0.171171 | 0.016219 | 0.050310 |

Downstream frozen action-decoder metrics from predicted motion:

| seed | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.056467 | 0.132502 | 0.016599 | 1.997458 | 0.245097 |
| 17 | 0.050788 | 0.123911 | 0.015971 | 2.065720 | 0.214798 |
| mean | 0.053628 | 0.128207 | 0.016285 | 2.031589 | 0.229947 |

## Comparison To Bounds

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero future motion | 0.001896 | 0.110711 | 0.199303 | 0.027867 | 2.619721 | 0.373234 |
| Gate 2 context-only prior | 0.001027 | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| Gate 2.2a DINO global visual prior | 0.000801 | 0.053628 | 0.128207 | 0.016285 | 2.031589 | 0.229947 |
| direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

Relative readout:

- future-motion MSE improves by `13.68%` over Gate 2.1.
- future-motion MSE improves by `21.96%` over Gate 2.
- downstream action MSE improves by `26.03%` over Gate 2.1.
- downstream action MSE improves by `18.76%` over direct context.
- Gate 2.2a closes `35.85%` of the direct-context to oracle-future-motion MSE
  gap.

## Interpretation

Gate 2.2a is the first learned future-motion prior that crosses the key
promotion boundary:

```text
learned future motion action MSE < direct context action MSE
```

This supports the central GeoMoCo-WM direction: visual grounding is not just an
extra modality; it supplies information that makes predicted future EEF motion
more executable under the frozen action-decoder interface.

The result is especially important because Gate 2.1 already included
suite/task metadata. The additional improvement therefore comes from visual
features beyond task identity.

## Limits

- This is still global-token grounding, not patch-level object/region grounding.
- No shuffled-vision control has been run yet, so this is a positive mechanism
  signal but not a complete anti-leakage/anti-correlation proof.
- The model is still deterministic and unimodal.
- The downstream evaluation is an action-decoder interface, not a closed-loop
  simulator success rate.

## Next Decision

Proceed to Gate 2.2b, but keep the attribution controls tight:

```text
Gate 2.2b: patch-token cross-attention grounding
Gate 2.2-control: shuffled visual features
Gate 2.2-camera: agentview-only vs eye-in-hand-only vs two-camera
then: visual-grounded GeoMoCo-cVAE
```

