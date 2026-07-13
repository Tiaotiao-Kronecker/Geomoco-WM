# Gate 2.4h-b Visual Phase/Event Probe

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.4h-b
- Purpose: test whether visual grounding, proprioception, and future motion
  predict the gripper transition labels from Gate 2.4h-a.

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
| horizon | 8 |
| split policy | episode |
| seeds | 7, 17 |

## Target

Five transition labels from Gate 2.4h-a:

```text
close_transition
mixed_transition
open_transition
sustain_close
sustain_open
```

Validation label counts:

```text
close_transition: 199
mixed_transition: 5
open_transition: 183
sustain_close: 1259
sustain_open: 1786
```

`mixed_transition` is extremely rare, so macro-F1 is a strict metric. Per-class
transition F1 should be inspected alongside macro-F1.

## Input Variants

```text
task_only
task_proprio
future_motion_only
proprio_future_motion
visual_only
visual_proprio
visual_proprio_future_motion
shuffled_visual_proprio_future_motion
```

The shuffled-visual branch keeps proprio, task, and future motion aligned but
permutes DINO visual features within batches. It is the main visual attribution
control.

## Code

```text
scripts/train_phase_event_probe.py
src/geomoco_wm/data/event_labels.py
tests/test_event_labels.py
```

## Commands

Example:

```bash
.venv/bin/python scripts/train_phase_event_probe.py \
  --input-variant visual_proprio_future_motion \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-dir outputs/phase_event_probe/gate2_4h_b_visual_proprio_future_motion_seed7 \
  --epochs 12 \
  --batch-size 256 \
  --device cuda \
  --seed 7 \
  --quiet
```

## Artifacts

Metrics/checkpoints are under:

```text
outputs/phase_event_probe/gate2_4h_b_<input_variant>_seed7/
outputs/phase_event_probe/gate2_4h_b_<input_variant>_seed17/
```

## Results

Mean over seed 7 and seed 17:

| input variant | accuracy | balanced acc | macro-F1 | close F1 | open F1 | sustain-close F1 | sustain-open F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task_only | 0.403061 | 0.230726 | 0.215345 | 0.113402 | 0.000000 | 0.382699 | 0.580623 |
| task_proprio | 0.626556 | 0.556063 | 0.421542 | 0.372290 | 0.363736 | 0.526582 | 0.806641 |
| future_motion_only | 0.675544 | 0.528531 | 0.442754 | 0.354974 | 0.358721 | 0.725552 | 0.762760 |
| proprio_future_motion | 0.745360 | 0.622805 | 0.509978 | 0.526389 | 0.370190 | 0.760295 | 0.875772 |
| visual_only | 0.893690 | 0.699496 | 0.630579 | 0.688070 | 0.596158 | 0.923715 | 0.944952 |
| visual_proprio | 0.888737 | 0.704846 | 0.625784 | 0.674754 | 0.594154 | 0.911827 | 0.948185 |
| visual_proprio_future_motion | 0.893803 | 0.704483 | 0.631385 | 0.691126 | 0.597560 | 0.922586 | 0.945653 |
| shuffled_visual_proprio_future_motion | 0.409505 | 0.198301 | 0.179985 | 0.000000 | 0.035398 | 0.420058 | 0.444468 |

## Interpretation

Gate 2.4h-b gives a strong positive visual-grounding result for phase/event
prediction.

Key observations:

- `task_only` is weak, so the probe is not explained by task identity alone.
- `task_proprio` and `future_motion_only` both contain useful event signal.
- `proprio_future_motion` improves over either alone, supporting the GeoMoCo
  phase/composition view.
- `visual_only` is already strong, which means DINO features contain current
  manipulation phase evidence.
- `visual_proprio_future_motion` is the strongest by macro-F1, though only
  slightly above `visual_only`.
- shuffled visual collapses below even task/proprio baselines, confirming the
  visual signal depends on correct alignment.

This is a stronger and more interpretable visual grounding result than simply
reporting a small action-MSE improvement. Vision helps identify the current
phase/event boundary, while future motion provides candidate transition
geometry.

## Caveats

- The split is episode-level, not held-out-task.
- `mixed_transition` has only 5 validation windows, so macro-F1 is harsh and
  noisy.
- `visual_only` being very strong could still include scene/task/episode style
  cues. The next probe should add held-out-task or task-family controls before
  treating this as generalization.

## Decision

Gate 2.4h-b passes as a phase/event probe.

Next mainline step:

```text
Gate 2.4h-c: cVAE sample event-alignment analysis
```

This should test whether cVAE samples selected by prior mean / ScoreNet /
oracle best-of-K differ in transition-event alignment.

Do not yet train an event-aware ScoreNet. First verify whether generated
samples carry event alignment signal.

## Verification

```text
.venv/bin/python -m unittest tests.test_event_labels
.venv/bin/python -m compileall src scripts tests
.venv/bin/ruff check src scripts tests
```

All passed before the formal GPU runs.
