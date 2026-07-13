# Gate 2.5e Event-Aware Joint Readout Pilot

- Date: 2026-06-10
- Status: pilot completed
- Gate: Gate 2.5e-a / Gate 2.5e-b
- Purpose: test whether transition-event-aware readout can improve joint cVAE
  sample selection over the Gate 2.5d flat action-MSE scorer.

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
| windows | 16,518 |
| horizon | 8 |
| motion mode | `future_delta_gripper` |
| split policy | episode |
| cVAE samples | 16 |
| seed | 7 pilot |
| device | cuda |

## Code

```text
scripts/train_visual_cvae_sample_scorer.py
scripts/evaluate_cvae_event_alignment.py
tests/test_future_motion_predictor.py
```

New optional training path:

```text
--event-target-weight <w>
--event-hard-negative-weight <w_hn>
```

The event-hard-negative loss uses the action+event composite target as the
positive and chooses a negative candidate that has relatively low action error
but bad transition-event alignment. This targets the failure mode where a sample
looks action-plausible under flat MSE but closes/opens at the wrong phase.

## Commands

Event-aware positive target, `w=0.05`:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_5e_event_w005_k16_seed7 \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --event-target-weight 0.05 \
  --target-kind action \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7 \
  --quiet
```

Event-aware positive target, `w=0.10` used the same command with:

```text
--output-dir outputs/visual_cvae_sample_scorer/gate2_5e_event_w01_k16_seed7
--event-target-weight 0.1
```

Event hard-negative pilot:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_5e_event_hn_w005_hn01_k16_seed7 \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --event-target-weight 0.05 \
  --event-hard-negative-weight 0.1 \
  --target-kind action \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7 \
  --quiet
```

Event alignment was evaluated with:

```bash
.venv/bin/python scripts/evaluate_cvae_event_alignment.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --scorer-checkpoint <scorer-model.pt> \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/event_alignment/<run-name>.json \
  --num-samples 16 \
  --batch-size 128 \
  --device cuda \
  --seed 7 \
  --quiet
```

## Artifacts

```text
outputs/visual_cvae_sample_scorer/gate2_5e_event_w005_k16_seed7/
outputs/visual_cvae_sample_scorer/gate2_5e_event_w01_k16_seed7/
outputs/visual_cvae_sample_scorer/gate2_5e_event_hn_w005_hn01_k16_seed7/
outputs/event_alignment/gate2_5e_event_w005_seed7_k16.json
outputs/event_alignment/gate2_5e_event_w01_seed7_k16.json
outputs/event_alignment/gate2_5e_event_hn_w005_hn01_seed7_k16.json
```

## Results

Seed 7 pilot:

| method | action MSE | action rank | event acc | macro-F1 | transition acc | step@1 | event rank | event error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.5d flat | 0.045230 | 5.981643 | 0.893065 | 0.630958 | 0.622739 | 0.240310 | 1.746503 | 1.841783 |
| event `w=0.05` | 0.045455 | 5.998543 | 0.893357 | 0.639819 | 0.602067 | 0.250646 | 1.753788 | 1.821387 |
| event `w=0.10` | 0.045710 | 6.038462 | 0.898019 | 0.629458 | 0.612403 | 0.271318 | 1.722902 | 1.764277 |
| event HN `w=0.05, hn=0.10` | 0.047043 | 6.102273 | 0.904429 | 0.639155 | 0.645995 | 0.284238 | 1.708333 | 1.714452 |

## Interpretation

Gate 2.5e is a useful negative pilot.

Adding transition-event supervision does change the readout in the intended
direction: event accuracy, transition timing, and event alignment error improve
as the event signal becomes stronger. However, deployable action MSE worsens
monotonically relative to the Gate 2.5d flat scorer on the same seed.

This means the current event signal is real but not yet action-compatible. In
the present candidate set, selecting the most event-aligned sample can move the
decoder away from the action-useful sample. The next mainline should therefore
not be a larger event-weight sweep. It should improve the joint sample/action
interface or train a readout that predicts action regret directly while using
event labels as diagnostics rather than as the main selection objective.

## Decision

Do not promote Gate 2.5e-a/b as the main scorer.

Next mainline options:

1. Add action-regret-aware readout features/targets that predict the frozen
   decoder's downstream regret directly, with event metrics reported as
   diagnostics.
2. Add a stronger temporal readout model over candidate action chunks instead
   of a shallow per-sample MLP scorer.
3. Revisit cVAE training so samples jointly preserve EEF geometry and gripper
   event timing before more readout engineering.

## Verification

```text
.venv/bin/python -m compileall scripts/train_visual_cvae_sample_scorer.py scripts/evaluate_visual_cvae_sample_scorer.py tests/test_future_motion_predictor.py
.venv/bin/ruff check scripts/train_visual_cvae_sample_scorer.py scripts/evaluate_visual_cvae_sample_scorer.py tests/test_future_motion_predictor.py
.venv/bin/python -m unittest tests.test_future_motion_predictor
```

All passed before the hard-negative pilot run.

