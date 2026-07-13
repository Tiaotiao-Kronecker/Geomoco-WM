# Gate 3.1b Event Mode Probe

## Purpose

Gate 3.1a showed that gripper-transition event modes are measurable. Gate 3.1b
tests whether those event modes are predictable from context and aligned visual
features before changing the cVAE.

This is a probe, not the final method.

## Inputs

```text
windows:
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl

event-mode labels:
outputs/event_modes/gate3_1a_event_modes_2files.json

real visual cache:
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5

shuffled visual cache:
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5
```

## Class Set

Stable 8-class event-mode set:

```text
sustain_open::none
sustain_close::none
transition_close::early
transition_close::middle
transition_close::late
transition_open::early
transition_open::middle
transition_open::late
```

Rare mixed-transition modes were dropped:

```text
mixed_transition::early = 10 windows
mixed_transition::middle = 4 windows
```

## Commands

Task/proprio:

```bash
.venv/bin/python scripts/train_event_mode_probe.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/event_mode_probe/gate3_1b_task_proprio_seed{7,17} \
  --input-variant task_proprio \
  --epochs 15 \
  --batch-size 256 \
  --seed {7,17} \
  --device cuda \
  --quiet
```

Real visual/proprio:

```bash
.venv/bin/python scripts/train_event_mode_probe.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17} \
  --input-variant visual_proprio \
  --epochs 15 \
  --batch-size 256 \
  --seed {7,17} \
  --device cuda \
  --quiet
```

Shuffled visual/proprio:

```bash
.venv/bin/python scripts/train_event_mode_probe.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5 \
  --output-dir outputs/event_mode_probe/gate3_1b_shuffled_visual_proprio_seed{7,17} \
  --input-variant visual_proprio \
  --epochs 15 \
  --batch-size 256 \
  --seed {7,17} \
  --device cuda \
  --quiet
```

## Results

| variant | seed | accuracy | balanced acc | macro-F1 | transition F1 | timing acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 7 | 0.592063 | 0.427355 | 0.304718 | 0.416970 | 0.387435 |
| task/proprio | 17 | 0.626039 | 0.413546 | 0.307720 | 0.429113 | 0.336364 |
| real visual/proprio | 7 | 0.804202 | 0.535542 | 0.434598 | 0.596581 | 0.426702 |
| real visual/proprio | 17 | 0.827639 | 0.583785 | 0.462885 | 0.603298 | 0.490909 |
| shuffled visual/proprio | 7 | 0.233149 | 0.126950 | 0.090514 | 0.192268 | 0.099476 |
| shuffled visual/proprio | 17 | 0.205602 | 0.131695 | 0.089497 | 0.190787 | 0.218182 |

Mean across seeds:

| variant | accuracy | balanced acc | macro-F1 | transition F1 | timing acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 0.609051 | 0.420450 | 0.306219 | 0.423041 | 0.361899 |
| real visual/proprio | 0.815921 | 0.559663 | 0.448741 | 0.599939 | 0.458805 |
| shuffled visual/proprio | 0.219375 | 0.129323 | 0.090006 | 0.191527 | 0.158829 |

## Artifacts

```text
outputs/event_mode_probe/gate3_1b_task_proprio_seed7/metrics.json
outputs/event_mode_probe/gate3_1b_task_proprio_seed17/metrics.json
outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/metrics.json
outputs/event_mode_probe/gate3_1b_visual_proprio_seed17/metrics.json
outputs/event_mode_probe/gate3_1b_shuffled_visual_proprio_seed7/metrics.json
outputs/event_mode_probe/gate3_1b_shuffled_visual_proprio_seed17/metrics.json
```

## Interpretation

Gate 3.1b passes.

Aligned visual grounding strongly improves event-mode prediction over
task/proprio alone:

```text
macro-F1: 0.306219 -> 0.448741
transition F1: 0.423041 -> 0.599939
timing accuracy: 0.361899 -> 0.458805
```

The shuffled visual control collapses:

```text
macro-F1: 0.090006
transition F1: 0.191527
```

So the event-mode signal is not merely a dataset prior or label artifact.
Correct visual alignment carries useful close/open timing information.

## Decision

Proceed to Gate 3.1c event-conditioned cVAE.

The first cVAE branch should separate:

```text
oracle event mode conditioning = upper bound
predicted event mode mixture = deployable route
shuffled event mode conditioning = control
```

Do not claim solved mode selection from this probe. This result only justifies
adding event-mode structure to the future-motion prior.
