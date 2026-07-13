# Gate 2.4h GeoMoCo Phase/Event Probe Plan

- Date: 2026-06-09
- Status: active
- Position in mainline: after Gate 2.4g hard-negative readout, before adding a
  stronger diffusion/flow action head.

## Purpose

Gate 2.4d showed that cVAE sample readout is useful but weak. Gates 2.4e/2.4f
showed that simply swapping flat action MSE for `SE(3)` / gripper metrics is
not enough. Gate 2.4g showed that naive structured-score hard negatives also
regress.

Gate 2.4h reframes the next readout question around GeoMoCo's intended
strength:

```text
Does the model capture manipulation phase/composition timing?
```

The observable probe is gripper/contact/event timing:

```text
approach -> align -> close/contact -> transport -> release
```

## Main Design

Event labels are weak labels derived from existing streams, not human labels:

```text
future action chunk -> close / open / hold / mixed
future action chunk -> close_step / open_step
gripper state width deltas -> close-sign audit
future EEF deltas -> motion waypoint / phase geometry
DINO visual tokens -> current-state grounding evidence
```

Visual information is not used to create event labels. It is used to test
whether phase/composition timing can be inferred better from grounded state.

## Gate Sequence

### Gate 2.4h-a: GT Gripper/Event Label Audit

Inputs:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Outputs:

```text
outputs/event_audits/gate2_4h_gripper_events_2files.json
outputs/event_audits/gate2_4h_gripper_events_2files.md
```

Questions:

- Which gripper action sign corresponds to close?
- How often do close/open/hold/mixed events occur?
- Are event steps distributed or concentrated?
- Are event timings task-specific shortcuts?
- Do event types correlate with future EEF displacement?

### Gate 2.4h-b: Visual Phase/Event Probe

Train tiny probe heads for:

```text
event_type
has_close / has_open
close_step / open_step
phase label
```

Compare input variants:

```text
task/proprio only
visual only
future motion only
proprio + future motion
visual + proprio
visual + proprio + future motion
shuffled visual + proprio + future motion
```

Pass signal:

```text
visual + proprio + future motion
  > proprio + future motion
  > task/proprio only
```

Shuffled visual must not improve.

### Gate 2.4h-c: cVAE Sample Event Alignment

Status: completed.

For each cVAE sample:

```text
sampled_future_motion_k
  -> frozen action decoder
  -> decoded_action_k
  -> candidate event label
```

Measure:

```text
event_type match
close/open timing error
event-aligned sample action MSE
event-aligned sample oracle rank
flat ScoreNet selected event alignment
```

Result:

```text
prior mean event acc: 0.836666
event oracle best-of-K event acc: 0.849049
prior mean transition step within 1: 0.179168
event oracle transition step within 1: 0.219359
ScoreNet transition step within 1: 0.175082
```

Interpretation: samples contain some event-aligned candidates, but transition
timing coverage is weak and the current flat ScoreNet does not explicitly select
event-aligned samples. Gate 2.4h-d should be a minimal controlled event-aware
readout diagnostic, not a large method jump.

### Gate 2.4h-d: Event-Aware ScoreNet

Status: completed as a minimal diagnostic; not promoted.

Tested:

```text
loss = flat_action_rank_loss + 0.1 * event_alignment_rank_loss
```

Result:

```text
flat ScoreNet action MSE: 0.040201
event w=0.1 action MSE: 0.040255
flat ScoreNet transition step within 1: 0.175082
event w=0.1 transition step within 1: 0.173790
```

Interpretation: a weak event ranking auxiliary is not sufficient. Promotion
still uses deployable action MSE and oracle rank, not event loss alone.

Next: move upstream to an event-fidelity interface audit before more readout
engineering.

## Anti-Overfitting Controls

- shuffled event labels;
- shuffled visual tokens;
- task-only event predictor;
- time-index-only event predictor;
- episode-level split;
- later held-out task split if the probe looks promising.

## Mainline Decision

Gate 2.4h is required as a diagnostic before more scorer engineering. It is not
required to become a final method module. If event labels are noisy or shortcut
dominated, skip event-aware scorer and move to stronger action decoders or
closed-loop evaluation.
