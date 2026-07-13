# Gate 2.4h-c cVAE Sample Event Alignment

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.4h-c
- Purpose: test whether calibrated visual cVAE samples contain close/open
  transition candidates, and whether existing sample readouts select them.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| val windows | 3,432 / 3,252 |
| horizon | 8 |
| split policy | episode |
| cVAE samples | 16 |
| seeds | 7, 17 |
| device | cuda |

Validation event counts:

```text
seed7:  close_transition 199, mixed_transition 5, open_transition 183, sustain_close 1259, sustain_open 1786
seed17: close_transition 170, mixed_transition 3, open_transition 160, sustain_close 1143, sustain_open 1776
```

## Code

```text
scripts/evaluate_cvae_event_alignment.py
src/geomoco_wm/data/event_labels.py
tests/test_cvae_event_alignment.py
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/evaluate_cvae_event_alignment.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --scorer-checkpoint outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed7/model.pt \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/event_alignment/gate2_4h_c_seed7_k16.json \
  --num-samples 16 \
  --batch-size 128 \
  --device cuda \
  --quiet
```

Seed 17 uses the matching `seed17` cVAE and ScoreNet checkpoints and writes:

```text
outputs/event_alignment/gate2_4h_c_seed17_k16.json
```

## Mean Results

Mean over seed 7 and seed 17:

| readout | event acc | macro-F1 | transition acc | transition step within 1 | pred transition rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior_mean | 0.836666 | 0.557594 | 0.421782 | 0.179168 | 0.087995 |
| random_sample_mean | 0.835440 | 0.554618 | 0.416745 | 0.172950 | 0.088369 |
| event_oracle_best | 0.849049 | 0.576540 | 0.452092 | 0.219359 | 0.084733 |
| flat_action_oracle_best | 0.841919 | 0.557707 | 0.412110 | 0.161534 | 0.085308 |
| se3_gripper_oracle_best | 0.838495 | 0.553209 | 0.405231 | 0.161324 | 0.085899 |
| scorer_argmax | 0.839062 | 0.553349 | 0.408688 | 0.175082 | 0.087202 |

Sample coverage over `K=16`:

| metric | mean |
| --- | ---: |
| any event type match | 0.849049 |
| any transition type match | 0.452092 |
| any transition step exact | 0.125044 |
| any transition step within 1 | 0.219359 |
| any transition step within 2 | 0.317096 |
| best event alignment error | 2.202891 |

Selected event-oracle rank:

| selector | event-oracle rank | event alignment error |
| --- | ---: | ---: |
| flat_action_oracle_best | 1.172401 | 2.351765 |
| se3_gripper_oracle_best | 1.190819 | 2.394400 |
| scorer_argmax | 1.256558 | 2.401051 |

## Interpretation

Gate 2.4h-c is a useful but mixed diagnostic.

Positive signal:

- cVAE samples do contain some event-aligned candidates: event oracle best-of-K
  improves event accuracy from `0.836666` to `0.849049`.
- Transition type coverage rises to `0.452092`, so sampled futures are not
  completely blind to close/open phase changes.

Main bottleneck:

- Transition timing coverage remains weak. Only `0.219359` of GT transition
  windows have a sample with the correct event type and step within 1.
- Random samples do not beat prior mean, so stochasticity alone does not give a
  useful event readout.
- Gate 2.4d ScoreNet does not select event-aligned samples better than the
  non-deployable event oracle. Its event metrics are close to prior mean and
  below flat action oracle.

The practical conclusion is:

```text
The current cVAE has limited event-aligned sample coverage, and the current
flat ScoreNet does not explicitly read out close/open transition timing.
```

## Decision

Do not immediately train a full event-aware scorer as if event supervision were
solved. The next mainline should first add a minimal Gate 2.4h-d event-aware
readout diagnostic:

```text
flat action-rank loss
  + weak transition-type auxiliary signal
  + careful promotion by action MSE and event metrics
```

If that does not improve both action readout and transition alignment, move the
main effort from readout engineering to improving the cVAE/action-decoder event
interface itself.

## Verification

```text
.venv/bin/python -m unittest tests.test_event_labels tests.test_cvae_event_alignment
.venv/bin/python -m compileall scripts/evaluate_cvae_event_alignment.py tests/test_cvae_event_alignment.py src/geomoco_wm/data/event_labels.py
.venv/bin/ruff check scripts/evaluate_cvae_event_alignment.py tests/test_cvae_event_alignment.py tests/test_event_labels.py src/geomoco_wm/data/event_labels.py
```

All passed before the formal CUDA runs.
