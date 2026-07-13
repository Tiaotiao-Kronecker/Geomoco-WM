# Gate 2.1 Suite/Task-Conditioned Future-Motion Prior

- Date: 2026-06-07
- Status: completed
- Gate: Gate 2.1
- Purpose: test whether adding categorical suite/task metadata to the
  deterministic future-motion prior makes predicted future EEF motion useful
  for the frozen oracle action-decoder interface.

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

Suite/task one-hot vocabulary:

| field | value |
| --- | ---: |
| condition mode | `suite_task` |
| encoding | one-hot |
| condition dim | 8 |
| vocab source | full dataset metadata labels |

Using full metadata labels for the categorical vocabulary is not motion/action
label leakage: the task and suite names are known problem metadata, not future
trajectory targets.

## Model And Training Config

```text
script: scripts/train_future_motion_predictor.py
model: FutureMotionPredictor(context + suite_task one-hot -> flattened future_delta)
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
weight decay: 0
split policy: episode
seed(s): 7, 17
device: cpu
downstream decoder: frozen Gate 1.6 geodesic oracle future-motion ActionDecoder
```

Code changes for this gate:

- `src/geomoco_wm/models/future_motion_predictor.py` now supports optional
  `conditioning_dim` and concatenates conditioning vectors with context.
- `scripts/train_future_motion_predictor.py` now supports
  `--condition-on none|suite|task|suite_task`.
- `tests/test_future_motion_predictor.py` covers conditioned predictor shape
  and missing-conditioning errors.

Default behavior remains unchanged because `--condition-on none` is still the
default.

## Commands

Dry-run:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --condition-on suite_task \
  --dry-run
```

Seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/future_motion_predictor/gate2_1_suite_task_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 7 \
  --device cpu \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Seed 17:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/future_motion_predictor/gate2_1_suite_task_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed 17 \
  --device cpu \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Results

Future-motion validation metrics:

| seed | val MSE | val MAE | val L2 | trans L2 | orient coord L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.000931 | 0.018338 | 0.188556 | 0.021643 | 0.053448 |
| 17 | 0.000926 | 0.018296 | 0.187197 | 0.022284 | 0.052692 |
| mean | 0.000929 | 0.018317 | 0.187877 | 0.021964 | 0.053070 |

Downstream frozen action-decoder metrics from predicted motion:

| seed | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.075907 | 0.159165 | 0.020646 | 2.081804 | 0.304188 |
| 17 | 0.069094 | 0.151981 | 0.021037 | 2.131178 | 0.254576 |
| mean | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |

## Comparison To Gate 2

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero future motion | 0.001896 | 0.110711 | 0.199303 | 0.027867 | 2.619721 | 0.373234 |
| Gate 2 context-only prior | 0.001027 | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

Relative readout:

- future-motion MSE improves by `9.59%` over Gate 2 context-only prior.
- future-motion MSE improves by `51.03%` over zero future motion.
- downstream action MSE improves by `10.81%` over Gate 2 context-only prior.
- downstream action MSE improves by `34.51%` over zero future motion.
- downstream action MSE remains `9.83%` worse than direct context.

## Interpretation

Suite/task conditioning is useful but not sufficient.

It improves both direct future-motion prediction and downstream action decoding
relative to the Gate 2 context-only prior, so task metadata carries real signal.
However, the predicted motion still does not beat the direct-context action
decoder. Therefore this is not yet an effective policy intermediate variable.

The current ordering is:

```text
zero future motion > Gate 2 learned motion > Gate 2.1 learned motion > direct context > oracle future motion
```

Lower is better, so the learned prior has moved in the right direction, but it
has not crossed the key promotion boundary:

```text
learned future motion action MSE < direct context action MSE
```

## Limits

- This is still a deterministic MLP prior, not a multimodal cVAE.
- It uses task/suite metadata only, not visual grounding.
- It predicts future EEF coordinate deltas, not object state, contact state, or
  gripper intent.
- The run used CPU because the default restricted execution context does not
  expose CUDA; this is acceptable for the small MLP gate but should be upgraded
  for DINO/cVAE/full-suite training.

## Next Decision

Do not promote Gate 2.1 to the main method.

Next mainline should add information that is closer to execution value:

```text
Gate 2.2: visual grounding token / DINO feature conditioning
Gate 2.3: action-aware auxiliary loss or joint prior-decoder training
Gate 2.4: separate gripper/contact prediction branch
```

