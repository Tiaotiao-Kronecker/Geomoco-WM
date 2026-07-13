# Gate 1.6 Two-File Oracle Action Decoder Geodesic Replacement

- Date: 2026-06-06
- Status: completed
- Gate: `Gate 1.6`
- Purpose: rerun the two-file four-suite oracle action-decoder gate after the
  LIBERO action semantics audit, replacing the older normalized-only SE(3)
  table with physical translation and SO(3) geodesic rotation metrics.

## Measurement Contract

Action semantics were confirmed in:

```text
docs/experiments/runs/2026-06-06_action_semantics_audit_geodesic_metrics.md
```

Canonical LIBERO action interpretation:

```text
action dim: 7
translation dims: 0:3, normalized delta scaled by 0.05 meters
rotation dims: 3:6, normalized rotvec scaled by 0.5 radians
gripper dim: 6
rotation metric: SO(3) geodesic between Exp(pred_rotvec_scaled) and Exp(gt_rotvec_scaled)
```

This record keeps normalized flat/split metrics for comparability, but the
primary geometric readings are now:

```text
val_translation_m_l2
val_rotation_geodesic_rad
val_rotation_geodesic_deg
```

## Dataset Slice

Source:

```text
/home/user/dataset/libero_official
```

Export output:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 task files | 8 |
| demos / episodes | 400 |
| tasks | 8 |
| windows | 16,518 |
| frames | 69,073 |
| dropped short episodes | 0 |
| exporter warnings | 0 |
| combined `windows.jsonl` size | 68M |

Per-suite windows:

| suite | windows |
| --- | ---: |
| `libero_spatial` | 2,546 |
| `libero_object` | 3,602 |
| `libero_goal` | 4,125 |
| `libero_10` | 6,245 |

## Training Config

Common settings:

```text
script: scripts/train_oracle_action_decoder.py
model: MLP ActionDecoder
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
split-by: episode
device: cuda
seeds: 7, 17
environment: /home/user/projects/Geomoco-WM/.venv
```

Branches:

| branch | command mode | context dim | motion dim | action dim | horizon |
| --- | --- | ---: | ---: | ---: | ---: |
| direct context | `--motion-mode none` | 15 | 0 | 7 | 8 |
| oracle future motion | `--motion-mode future_delta` | 15 | 48 | 7 | 8 |

CUDA visibility was confirmed in the project-local environment:

```text
torch: 2.12.0+cu130
torch.cuda.is_available(): True
device: NVIDIA GeForce RTX 5090
```

## Commands

Seed 7 direct:

```bash
.venv/bin/python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --motion-mode none \
  --split-by episode \
  --seed 7 \
  --device cuda
```

Seed 7 future:

```bash
.venv/bin/python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 7 \
  --device cuda
```

Seed 17 direct:

```bash
.venv/bin/python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --motion-mode none \
  --split-by episode \
  --seed 17 \
  --device cuda
```

Seed 17 future:

```bash
.venv/bin/python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 17 \
  --device cuda
```

## Clean Replacement Table

Final validation metrics:

| seed | branch | val MSE | val MAE | trans L2 (m) | rot geo (deg) | rot geo (rad) | gripper MSE | SE(3) MSE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | direct context | 0.068109 | 0.148911 | 0.018819 | 2.182476 | 0.038091 | 0.278115 | 0.033109 |
| 7 | oracle future motion | 0.033542 | 0.082386 | 0.007464 | 1.014279 | 0.017703 | 0.199718 | 0.005846 |
| 17 | direct context | 0.063910 | 0.145336 | 0.019228 | 2.284825 | 0.039878 | 0.226975 | 0.036733 |
| 17 | oracle future motion | 0.029407 | 0.076629 | 0.007467 | 1.081786 | 0.018881 | 0.169648 | 0.006033 |

Relative validation reduction from direct context to oracle future motion:

| seed | MSE | MAE | trans L2 (m) | rot geo | gripper MSE | SE(3) MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 50.75% | 44.67% | 60.34% | 53.53% | 28.19% | 82.34% |
| 17 | 53.99% | 47.27% | 61.16% | 52.65% | 25.26% | 83.58% |
| mean | 52.32% | 45.96% | 60.76% | 53.08% | 26.87% | 82.99% |

Mean validation metrics across seeds:

| branch | val MSE | val MAE | trans L2 (m) | rot geo (deg) | rot geo (rad) | gripper MSE | SE(3) MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct context | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.038985 | 0.252545 | 0.034921 |
| oracle future motion | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.018292 | 0.184683 | 0.005939 |

## Artifacts

```text
outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed7/metrics.json
outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/metrics.json
outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed17/metrics.json
outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/metrics.json
outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed7/model.pt
outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed17/model.pt
outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Interpretation

This rerun replaces the older Gate 1.6 metric table without changing the data
slice, model, split policy, or seeds. The old flat MSE/MAE values are
reproduced within expected deterministic tolerance, and the new metrics make
the geometric gap physically readable:

```text
translation action error: 0.0190m -> 0.00747m mean L2
rotation action error: 2.23deg -> 1.05deg mean SO(3) geodesic
```

The oracle future-motion interface remains strongly positive. The largest
relative gains remain in geometric action recovery, while gripper gains are
smaller. This supports the next mainline step:

```text
learned future-motion prior -> same action-decoder gate
```

The first learned prior should be judged against these two fixed bounds:

```text
direct context lower bound
oracle future-motion upper/interface bound
```

## Next Step

Start Gate 2 with a learned future EEF-delta predictor on the same
2-files-per-suite slice. The first target should be deterministic
`context/proprio -> future_delta`, evaluated by both future-motion prediction
metrics and downstream action-decoder metrics.
