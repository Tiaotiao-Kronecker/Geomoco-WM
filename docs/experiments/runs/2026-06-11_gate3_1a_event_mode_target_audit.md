# Gate 3.1a Event Mode Target Audit

## Purpose

Gate 3.0c showed that the action head benefits from aligned sample-set
diversity, while shuffled generic diversity hurts. Gate 3.1a materializes the
first explicit mode target for that aligned diversity:

```text
event_mode = normalized gripper transition type + timing bin
```

The labels are weak action-derived proxies. They are not semantic contact
labels.

## Inputs

```text
windows:
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl

slice:
2 files per LIBERO suite, all demos, horizon 8, episode-level split
```

## Command

```bash
.venv/bin/python scripts/audit_event_modes.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-md outputs/event_modes/gate3_1a_event_modes_2files.md \
  --train-ratio 0.8 \
  --split-by episode \
  --seed 7 \
  --min-class-count 50
```

## Artifacts

```text
outputs/event_modes/gate3_1a_event_modes_2files.json
outputs/event_modes/gate3_1a_event_modes_2files.md
```

The JSON artifact includes per-window `event_mode` labels for later probes and
cVAE conditioning.

## Label Contract

Raw transition labels from `event_labels.py` are normalized as:

| raw label | Gate 3.1 event type |
| --- | --- |
| `close_transition` | `transition_close` |
| `open_transition` | `transition_open` |
| `mixed_transition` | `mixed_transition` |
| `sustain_close` | `sustain_close` |
| `sustain_open` | `sustain_open` |
| `hold` | `hold` |

For horizon 8, timing bins are:

| bin | steps |
| --- | --- |
| `early` | 0, 1, 2 |
| `middle` | 3, 4, 5 |
| `late` | 6, 7 |
| `none` | no transition step |

## Results

| event mode | count | fraction |
| --- | ---: | ---: |
| `sustain_open::none` | 8938 | 0.5411 |
| `sustain_close::none` | 5881 | 0.3560 |
| `transition_close::middle` | 339 | 0.0205 |
| `transition_close::early` | 329 | 0.0199 |
| `transition_open::middle` | 315 | 0.0191 |
| `transition_open::early` | 292 | 0.0177 |
| `transition_close::late` | 210 | 0.0127 |
| `transition_open::late` | 200 | 0.0121 |
| `mixed_transition::early` | 10 | 0.0006 |
| `mixed_transition::middle` | 4 | 0.0002 |

Event type totals:

| event type | count | fraction |
| --- | ---: | ---: |
| `sustain_open` | 8938 | 0.5411 |
| `sustain_close` | 5881 | 0.3560 |
| `transition_close` | 878 | 0.0532 |
| `transition_open` | 807 | 0.0489 |
| `mixed_transition` | 14 | 0.0008 |

Transition step histogram:

| step | count |
| --- | ---: |
| 0 | 208 |
| 1 | 225 |
| 2 | 198 |
| 3 | 205 |
| 4 | 214 |
| 5 | 239 |
| 6 | 201 |
| 7 | 209 |

Train/validation balance:

| split | windows | modes present |
| --- | ---: | ---: |
| train | 13086 | 10 |
| validation | 3432 | 10 |

Warnings:

```text
2 event modes have fewer than 50 windows.
```

The rare modes are `mixed_transition::early` and
`mixed_transition::middle`.

## Interpretation

Gate 3.1a passes the target-readiness check.

The useful transition classes are measurable: close/open transitions have
hundreds of windows each across early/middle/late bins. Train and validation
splits both contain all event modes. The transition-step histogram is not
dominated by step 0, so the combined label does not reproduce the old
command-state shortcut failure.

The mixed-transition classes are too rare for independent promotion. They
should remain diagnostic or be merged into a coarse `transition_mixed/other`
bucket if a later classifier needs stable class balance.

## Decision

Proceed to Gate 3.1b event-mode probe baseline.

Use the 8 stable deployable modes as the main class set:

```text
sustain_open::none
sustain_close::none
transition_close::{early,middle,late}
transition_open::{early,middle,late}
```

Keep mixed-transition labels in artifacts, but do not let them drive the main
training objective unless explicitly using class weighting or a merged rare
class.

## Next Step

Train a lightweight event-mode probe with real-vs-shuffled visual controls:

```text
context/task/proprio -> event_mode
visual + context/task/proprio -> event_mode
sample-summary + context/task/proprio -> event_mode
```

Promotion criterion for Gate 3.1b:

```text
real visual > shuffled visual, especially on transition-vs-sustain and
transition timing bins.
```
