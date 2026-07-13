# Gate 2 Deterministic Future-Motion Prior

- Date: 2026-06-06
- Status: completed
- Gate: `Gate 2`
- Purpose: train the first learned future-motion prior on the same
  2-files-per-suite LIBERO slice and evaluate whether predicted future EEF
  motion can sit between the direct-context lower bound and oracle future-motion
  upper/interface bound.

## Code Changes

New model:

```text
src/geomoco_wm/models/future_motion_predictor.py
```

New metrics:

```text
src/geomoco_wm/metrics/motion_metrics.py
```

New training / evaluation script:

```text
scripts/train_future_motion_predictor.py
```

New tests:

```text
tests/test_future_motion_predictor.py
```

Model:

```text
FutureMotionPredictor(context -> flattened future_delta)
context dim: 15
motion dim: 48
horizon: 8
future EEF step dim: 6
architecture: MLP, hidden dims 256,256
```

The script can optionally load a frozen oracle action decoder and report
downstream action metrics from predicted future motion:

```text
pred_motion = FutureMotionPredictor(context)
pred_actions = frozen ActionDecoder(context, pred_motion)
```

## Dataset Slice

Same slice as the Gate 1.6 geodesic replacement:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 task files | 8 |
| demos / episodes | 400 |
| windows | 16,518 |
| context dim | 15 |
| future motion dim | 48 |
| action dim | 7 |
| horizon | 8 |

Split:

```text
split-by: episode
train ratio: 0.8
seeds: 7, 17
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/future_motion_predictor/gate2_deterministic_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --split-by episode \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Seed 17:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/future_motion_predictor/gate2_deterministic_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --split-by episode \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Motion Prediction Results

Final validation future-motion metrics:

| seed | val MSE | val MAE | val L2 | trans L2 | orient coord L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.001014 | 0.019434 | 0.198509 | 0.024142 | 0.055423 |
| 17 | 0.001040 | 0.019646 | 0.199579 | 0.025428 | 0.055029 |
| mean | 0.001027 | 0.019540 | 0.199044 | 0.024785 | 0.055226 |

Zero-motion validation baseline on the same episode splits:

| seed | val MSE | val MAE | val L2 | trans L2 | orient coord L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.001884 | 0.022920 | 0.249807 | 0.029646 | 0.067051 |
| 17 | 0.001909 | 0.023149 | 0.251383 | 0.030098 | 0.067518 |
| mean | 0.001896 | 0.023034 | 0.250595 | 0.029872 | 0.067285 |

Interpretation: the deterministic predictor does learn nontrivial future
motion. Compared with zero future motion, it reduces mean future-motion MSE by
about 45.8%, mean chunk L2 by about 20.6%, translation L2 by about 17.0%, and
orientation-coordinate L2 by about 17.9%.

## Downstream Action Metrics

The predicted future motion was fed into the frozen same-seed oracle action
decoder from the Gate 1.6 geodesic replacement.

### How To Read The Four Branches

The word "it" in the diagnosis below refers to the learned future-motion prior:

```text
FutureMotionPredictor: current context/proprio -> predicted future EEF delta
```

The downstream action metric is not measuring future-motion prediction directly.
It measures the actions produced after a branch-specific motion input is fed
into an action decoder:

| branch | action-decoder input | what it tests |
| --- | --- | --- |
| direct context | current context only, no future-motion input | Lower-bound policy interface: can the action decoder predict actions directly from current state? |
| zero future motion | current context plus an all-zero future-motion vector | Sanity baseline: if the future-motion channel contains no useful information, how bad is the frozen oracle decoder? |
| learned future motion | current context plus `FutureMotionPredictor(context)` | Whether the learned prior is useful as a policy/action intermediate variable. |
| oracle future motion | current context plus GT future EEF delta from the dataset | Upper/interface bound: how good can this action-decoder route be when future motion is perfect? |

Lower action MSE is better. The ideal ordering for a useful learned prior would
be:

```text
zero future motion > direct context > learned future motion > oracle future motion
```

The observed ordering is:

```text
zero future motion > learned future motion > direct context > oracle future motion
```

So the learned future-motion prior is stronger than an empty future-motion
channel, but it is still not useful enough to beat direct action prediction
from context. This is the precise meaning of:

```text
future-motion MSE improves, downstream action value does not yet improve.
```

Final validation action metrics:

| seed | branch | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | direct context | 0.068109 | 0.148911 | 0.018819 | 2.182476 | 0.278115 |
| 7 | learned future motion | 0.082237 | 0.166955 | 0.022282 | 2.142468 | 0.314740 |
| 7 | oracle future motion | 0.033542 | 0.082386 | 0.007464 | 1.014279 | 0.199718 |
| 17 | direct context | 0.063910 | 0.145336 | 0.019228 | 2.284825 | 0.226975 |
| 17 | learned future motion | 0.080345 | 0.166050 | 0.023184 | 2.237309 | 0.278810 |
| 17 | oracle future motion | 0.029407 | 0.076629 | 0.007467 | 1.081786 | 0.169648 |

Mean validation action metrics:

| branch | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct context | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| zero future motion | 0.110711 | 0.199303 | 0.027867 | 2.619721 | 0.373234 |
| learned future motion | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| oracle future motion | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

Interpretation:

- learned future motion is better than zero future motion in downstream action
  decoding;
- learned future motion does not beat the direct-context lower-bound action
  decoder;
- the learned predictor slightly improves rotation geodesic over direct
  context, but hurts translation, flat action MSE/MAE, and gripper metrics;
- the oracle future-motion gap remains large, so the interface still has
  headroom.

## Artifacts

```text
outputs/future_motion_predictor/gate2_deterministic_seed7/metrics.json
outputs/future_motion_predictor/gate2_deterministic_seed7/model.pt
outputs/future_motion_predictor/gate2_deterministic_seed17/metrics.json
outputs/future_motion_predictor/gate2_deterministic_seed17/model.pt
```

Smoke artifact:

```text
outputs/future_motion_predictor/gate2_smoke_1epoch/metrics.json
outputs/future_motion_predictor/gate2_smoke_1epoch/model.pt
```

## Verification

```bash
.venv/bin/python -m compileall src scripts tests
.venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check src/geomoco_wm/models/future_motion_predictor.py src/geomoco_wm/metrics/motion_metrics.py scripts/train_future_motion_predictor.py tests/test_future_motion_predictor.py
git diff --check
```

Results:

- `compileall`: passed
- `unittest`: 16 tests passed
- `ruff`: passed
- `git diff --check`: passed

## Interpretation

This Gate 2 deterministic baseline is intentionally simple and attribution
clean. It establishes that:

1. Current proprioceptive context contains some future-motion signal.
2. Future-motion prediction MSE alone is not enough to guarantee downstream
   action-decoder value.
3. Context-only deterministic prediction likely averages over multimodal
   futures and lacks task/visual grounding.
4. The frozen oracle action decoder is sensitive to distribution shift: it was
   trained on GT future motion, and predicted future motion can land off the
   decoder's useful manifold.

This is not a stop signal for GeoMoCo-WM. It is a useful negative/diagnostic
gate: the next prior must add conditioning or objectives that make predicted
motion useful for action, not only low-MSE.

## Next Step

Do not jump directly to cVAE claims from this deterministic baseline. The next
implementation should add at least one of:

```text
task/suite conditioning
RGB/DINO visual grounding token
trajectory-level temporal model instead of per-window MLP
action-aware auxiliary loss through a frozen or jointly trained decoder
gripper/contact auxiliary prediction
```

The cleanest next engineering step is likely:

```text
Gate 2.1: add task/suite embedding + stronger temporal predictor, rerun the same downstream action gate
```

Then promote to visual grounding / cVAE once the learned-prior output begins to
move action metrics between direct context and oracle future motion.
