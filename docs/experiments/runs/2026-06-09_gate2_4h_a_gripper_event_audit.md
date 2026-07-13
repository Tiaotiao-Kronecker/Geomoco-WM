# Gate 2.4h-a Gripper/Event Label Audit

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.4h-a
- Purpose: audit whether gripper-derived event labels are usable as a
  GeoMoCo phase/composition probe before training any event-aware scorer.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |

## Method

Two label modes were audited:

```text
command-state:
  label the gripper command state inside the future action chunk.

transition:
  compare the command at action_start - 1 with the future action chunk and
  label only sign changes into close/open.
```

The close sign was inferred from HDF5 gripper-width deltas:

```text
positive gripper command count: 22,392
negative gripper command count: 17,608
positive mean width delta: -0.0008597
negative mean width delta:  0.0007448
inferred close sign: +1
```

So for this LIBERO slice, positive gripper command corresponds to closing.

## Terminology

LIBERO actions are 7D:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

In this slice, the sign audit indicates:

```text
gripper >  0.5 -> close command
gripper < -0.5 -> open command
otherwise      -> hold / unclear
```

`gripper command state` asks what state the future chunk is already commanding.
For example:

```text
future gripper: [+1, +1, +1, +1, +1, +1, +1, +1]
label: close
```

This means the chunk is commanding close throughout the horizon. It does not
mean a close transition happens inside the chunk. The gripper may already have
been closing before the window starts.

`transition label` compares the previous gripper command to the future chunk.
For example:

```text
previous command: -1
future gripper:  [-1, -1, +1, +1, +1, +1, +1, +1]
label: close_transition
close_step: 2
```

This means the future rollout enters close at step 2. This is the phase boundary
we care about for GeoMoCo.

If the previous command is already close and the future chunk stays close:

```text
previous command: +1
future gripper:  [+1, +1, +1, +1, +1, +1, +1, +1]
label: sustain_close
```

So the difference is:

```text
command-state label:
  describes the control state inside the future chunk.

transition label:
  describes whether the future chunk contains a close/open phase transition.
```

For GeoMoCo-WM, transition labels are the better phase/composition probe because
the claim is about when manipulation phases change, not merely whether a
sustained close/open command is present.

## Commands

Command-state audit:

```bash
.venv/bin/python scripts/audit_gripper_events.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-json outputs/event_audits/gate2_4h_gripper_events_2files.json \
  --output-md outputs/event_audits/gate2_4h_gripper_events_2files.md \
  --label-mode command
```

Transition audit:

```bash
.venv/bin/python scripts/audit_gripper_events.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-md outputs/event_audits/gate2_4h_gripper_transitions_2files.md \
  --label-mode transition
```

## Artifacts

```text
outputs/event_audits/gate2_4h_gripper_events_2files.json
outputs/event_audits/gate2_4h_gripper_events_2files.md
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
outputs/event_audits/gate2_4h_gripper_transitions_2files.md
```

Code:

```text
src/geomoco_wm/data/event_labels.py
scripts/audit_gripper_events.py
tests/test_event_labels.py
```

## Results

Command-state labels:

| event | count | fraction |
| --- | ---: | ---: |
| close | 5,984 | 0.3623 |
| mixed | 1,497 | 0.0906 |
| open | 9,037 | 0.5471 |

Command-state first close step:

```text
step 0: 6,699
step 1:   128
step 2:   103
step 3:   109
step 4:   105
step 5:   127
step 6:   102
step 7:   108
```

Transition labels:

| event | count | fraction |
| --- | ---: | ---: |
| close_transition | 878 | 0.0532 |
| mixed_transition | 14 | 0.0008 |
| open_transition | 807 | 0.0489 |
| sustain_close | 5,881 | 0.3560 |
| sustain_open | 8,938 | 0.5411 |

Transition first close step:

```text
step 0: 105
step 1: 128
step 2: 103
step 3: 109
step 4: 106
step 5: 128
step 6: 104
step 7: 109
```

Suite-level transition counts:

| suite | close_transition | open_transition | sustain_close | sustain_open |
| --- | ---: | ---: | ---: | ---: |
| libero_10 | 311 | 280 | 2,135 | 3,515 |
| libero_goal | 117 | 109 | 790 | 3,108 |
| libero_object | 234 | 230 | 1,736 | 1,402 |
| libero_spatial | 216 | 188 | 1,220 | 913 |

Motion by event type:

| event | n | final trans L2 | path trans L2 |
| --- | ---: | ---: | ---: |
| close_transition | 878 | 0.028171 | 0.030994 |
| open_transition | 807 | 0.023821 | 0.028843 |
| sustain_close | 5,881 | 0.058454 | 0.060138 |
| sustain_open | 8,938 | 0.054669 | 0.056055 |

## Interpretation

Command-state labels are too coarse for phase timing. They mostly report the
state that is already active at the first future action step, creating strong
step-0 concentration and task/time shortcut risk.

Transition labels are much better aligned with the intended phase probe. They
isolate actual sign changes into close/open, remove the detected shortcut risk,
and create a smaller but meaningful set of phase-boundary windows.

This is important for GeoMoCo-WM: the useful observable event is not "the
gripper is currently commanded closed", but "the rollout proposes a transition
into close/open at this future step."

## Decision

Use transition labels, not command-state labels, for Gate 2.4h-b/c/d.

Next step:

```text
Gate 2.4h-b visual phase/event probe
```

The first probe should compare:

```text
task/proprio only
future motion only
visual + proprio
proprio + future motion
visual + proprio + future motion
shuffled visual + proprio + future motion
```

against the transition labels above.

## Verification

```text
.venv/bin/python -m unittest tests.test_event_labels
.venv/bin/python -m compileall src scripts tests
.venv/bin/ruff check src scripts tests
```

All passed before the full audit.
