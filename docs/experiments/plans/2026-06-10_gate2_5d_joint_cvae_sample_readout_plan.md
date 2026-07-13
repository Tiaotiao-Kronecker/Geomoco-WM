# Gate 2.5d Joint cVAE Sample Readout Plan

- Date: 2026-06-10
- Gate: Gate 2.5d
- Status: executed

## Objective

Train a lightweight readout over joint cVAE future-motion samples
(`future_delta_ee + future_gripper/event`) and test whether learned selection
can recover useful future samples without GT best-of-K selection.

## Main Question

Can a learned scorer select action-useful samples from the joint cVAE sample set
better than prior mean, while preserving gripper/event fidelity?

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

## Train / Eval Config

```text
script:
  scripts/train_visual_cvae_sample_scorer.py
  scripts/evaluate_visual_cvae_sample_scorer.py
  scripts/evaluate_cvae_event_alignment.py
scorer target:
  action
sample count:
  16
motion mode:
  future_delta_gripper
condition:
  suite_task
split policy:
  episode
seeds:
  7, 17
device:
  cuda for training/eval
```

## Pass Criteria

Promotion requires all of the following:

- scorer argmax action MSE beats the deterministic joint baseline `0.040688`;
- real visual scoring beats shuffled visual scoring;
- event readout does not collapse relative to prior mean;
- selected rank closes a meaningful fraction of the oracle gap.

## Stop Criteria

If the scorer only slightly improves over prior mean or regresses from the
deterministic joint baseline, do not promote it as the main branch. Move to
event/contact/executability-aware readout or a richer scorer target.

