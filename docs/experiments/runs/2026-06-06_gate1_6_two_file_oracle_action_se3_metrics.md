# Gate 1.6 Two-File Oracle Action Decoder With SE(3)-Aware Metrics

- Date: 2026-06-06
- Status: completed
- Gate: `Gate 1.6`
- Purpose: scale the oracle future-motion action-decoder gate from one file per
  LIBERO suite to two files per suite, and add SE(3)-aware action metric
  decomposition.

## Code Changes

Metrics implementation:

```text
scripts/train_oracle_action_decoder.py
```

Added final/history metrics:

```text
mse, mae
translation_mse, translation_mae, translation_l2
rotation_mse, rotation_mae, rotation_l2
se3_mse, se3_mae, se3_l2
gripper_mse, gripper_mae
```

Test:

```text
tests/test_oracle_action_decoder_metrics.py
```

Note: this is an SE(3)-aware split metric over 7D action chunks:

```text
translation: action dims 0:3
rotation: action dims 3:6
gripper: action dims 6:
se3: action dims 0:6
```

It is not yet a strict SO(3) geodesic metric.

## Dataset Slice

Source:

```text
/home/user/dataset/libero_official
```

Export output:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/
```

Export command:

```bash
python scripts/export_libero_windows.py \
  --all-libero-suites \
  --input-root /home/user/dataset/libero_official \
  --output-dir outputs/libero_windows/libero_all_suites_2files_all_demos_h8 \
  --context-len 2 \
  --horizon 8 \
  --stride 4 \
  --max-files-per-suite 2
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 files | 8 |
| demos / episodes | 400 |
| tasks | 8 |
| windows | 16,518 |
| frames | 69,073 |
| dropped short episodes | 0 |
| exporter warnings | 0 |
| combined `windows.jsonl` size | 68M |
| combined `episodes.jsonl` size | 364K |

Per-suite windows:

| suite | episodes | windows |
| --- | ---: | ---: |
| `libero_spatial` | 100 | 2,546 |
| `libero_object` | 100 | 3,602 |
| `libero_goal` | 100 | 4,125 |
| `libero_10` | 100 | 6,245 |

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
```

Branches:

| branch | command mode | context dim | motion dim | action dim | horizon |
| --- | --- | ---: | ---: | ---: | ---: |
| direct context | `--motion-mode none` | 15 | 0 | 7 | 8 |
| oracle future motion | `--motion-mode future_delta` | 15 | 48 | 7 | 8 |

## Commands

Seed 7 direct:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_2files_direct_seed7 \
  --motion-mode none \
  --split-by episode \
  --seed 7 \
  --device cuda
```

Seed 7 future:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_2files_future_seed7 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 7 \
  --device cuda
```

Seed 17 direct:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_2files_direct_seed17 \
  --motion-mode none \
  --split-by episode \
  --seed 17 \
  --device cuda
```

Seed 17 future:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_2files_future_seed17 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 17 \
  --device cuda
```

## Overall Results

| seed | branch | train windows | val windows | val MSE | val MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | direct context | 13,086 | 3,432 | 0.068109 | 0.148911 |
| 7 | oracle future motion | 13,086 | 3,432 | 0.033542 | 0.082386 |
| 17 | direct context | 13,266 | 3,252 | 0.063910 | 0.145336 |
| 17 | oracle future motion | 13,266 | 3,252 | 0.029407 | 0.076629 |

Overall relative validation improvement:

| seed | MSE reduction | MAE reduction |
| ---: | ---: | ---: |
| 7 | 50.75% | 44.67% |
| 17 | 53.99% | 47.27% |

## SE(3)-Aware Validation Reductions

| seed | translation MSE | translation MAE | rotation MSE | rotation MAE | SE(3) MSE | SE(3) MAE | gripper MSE | gripper MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 82.56% | 60.41% | 77.25% | 52.70% | 82.34% | 59.14% | 28.19% | 18.81% |
| 17 | 83.93% | 60.81% | 75.08% | 51.70% | 83.58% | 59.28% | 25.26% | 22.69% |

Final validation metric values:

| seed | branch | trans MSE | rot MSE | SE(3) MSE | grip MSE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | direct | 0.063488 | 0.002730 | 0.033109 | 0.278115 |
| 7 | future | 0.011070 | 0.000621 | 0.005846 | 0.199718 |
| 17 | direct | 0.070562 | 0.002904 | 0.036733 | 0.226975 |
| 17 | future | 0.011342 | 0.000724 | 0.006033 | 0.169648 |

## Artifacts

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/summary.json
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/oracle_action_decoder/libero_all_suites_2files_direct_seed7/metrics.json
outputs/oracle_action_decoder/libero_all_suites_2files_future_seed7/metrics.json
outputs/oracle_action_decoder/libero_all_suites_2files_direct_seed17/metrics.json
outputs/oracle_action_decoder/libero_all_suites_2files_future_seed17/metrics.json
```

## Interpretation

The oracle future-motion gate remains positive after scaling from one file per
suite to two files per suite. The improvement is stable across two
episode-level seeds.

The SE(3)-aware split is especially informative:

```text
future EEF motion strongly improves translation and rotation action recovery;
gripper/contact improves, but much less.
```

This supports the GeoMoCo-WM interface direction for geometric motion, while
also suggesting that gripper/contact should be modeled or supervised separately
instead of assuming EEF motion alone will solve it.

## Limits

This still does not prove visual grounding, learned future-motion prediction,
cVAE sampling quality, or closed-loop policy success.

The rotation metric is currently a 3D action-coordinate error. A stricter
SO(3)/SE(3) geodesic metric should be added once the exact action rotation
parameterization is confirmed.

## Next Decision

Proceed to a learned future-motion prior on the exported windows. The first
model should predict or sample future EEF deltas from context, then be evaluated
through the same action-decoder interface against direct context and oracle
future-motion bounds.

