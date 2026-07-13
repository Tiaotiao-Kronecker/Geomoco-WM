# Gate 3.9a Flow Calibration And Dataset Sufficiency Audit

## Purpose

Follow up the initial Gate 3.9a flow-matching replacement audit in two ways:

```text
1. Check whether the weak flow policy is mainly limited by too few Euler steps.
2. Audit whether the current 2-files-per-suite dataset slice has enough
   transition/open-close examples for the main bottleneck.
```

## Flow Eval-Only Calibration

Added eval-only support to:

```text
scripts/train_flow_matching_action_policy.py
```

This reuses a trained `model.pt` and changes only:

```text
eval_steps
num_eval_passes
```

No retraining is performed.

## Flow Results

Seed 7, 5-pass validation:

| branch | steps | overall MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_visual_flow | 8 | 0.059032 | 0.227686 | 0.037597 | 0.266500 |
| direct_visual_flow | 16 | 0.061379 | 0.235067 | 0.039305 | 0.271005 |
| direct_visual_flow | 32 | 0.062646 | 0.237015 | 0.040485 | 0.272976 |
| geomoco_flow | 8 | 0.058218 | 0.216756 | 0.038069 | 0.267212 |
| geomoco_flow | 16 | 0.061158 | 0.223223 | 0.040561 | 0.279509 |
| geomoco_flow | 32 | 0.062459 | 0.225643 | 0.041719 | 0.282421 |

Interpretation:

```text
Increasing Euler steps makes both branches worse.
```

The weak flow results are therefore not explained by too few sampling steps.
The first-order issue is more likely velocity-field/action-calibration quality,
objective mismatch, or the dataset/transition sparsity, not the integration
step count.

## Dataset Sufficiency Audit

Added:

```text
scripts/audit_dataset_sufficiency.py
```

Artifacts:

```text
outputs/dataset_audits/gate3_9a_2files_sufficiency_audit.json
outputs/dataset_audits/gate3_9a_2files_sufficiency_audit.md
```

Current 2-files-per-suite slice:

```text
windows:      16,518
episodes:    400
suites:      4
tasks:       8
source files: 8
```

Event coverage:

| event type | count | fraction |
| --- | ---: | ---: |
| mixed_transition | 14 | 0.000848 |
| sustain_close | 5,881 | 0.356036 |
| sustain_open | 8,938 | 0.541107 |
| transition_close | 878 | 0.053154 |
| transition_open | 807 | 0.048856 |

Total transition fraction:

```text
1,699 / 16,518 = 0.102857
```

Warnings:

```text
transition windows are sparse: 0.1029 of all windows
close transition count is below 1000: 878
open transition count is below 1000: 807
```

Task-level issue:

```text
open_the_middle_drawer_of_the_cabinet transition fraction = 0.000000
```

So the current task mix includes at least one task that contributes no
transition windows under the current event-label definition.

## Interpretation

The user's dataset-size concern is valid.

The current slice is useful for fast mechanism tests, but it is probably not
enough to train or fairly judge a stronger flow/diffusion-style policy on the
transition/open-close bottleneck:

```text
positive transition examples are only about 10.3% of windows;
close/open examples are each below 1000;
one task contributes zero transition labels;
the main failure metric is exactly this sparse transition subset.
```

This also explains why stronger decoders drift toward sustain/continuous-motion
quality: most windows are sustain windows, and transition supervision is a
small minority.

## Decision

Do not continue full Gate 3.9a controls on the current 2-files-per-suite slice.

Recommended next step:

```text
Build a larger or transition-enriched training/eval slice before continuing the
strong-policy replacement audit.
```

Minimal concrete options:

```text
1. export/use more LIBERO files per suite to raise close/open transition counts;
2. create a transition-balanced training sampler while keeping validation
   natural-distribution and transition-sliced metrics;
3. select tasks with nonzero transition coverage for the next strong-policy
   audit, while recording the task-selection rule;
4. only then rerun direct_visual_flow vs geomoco_flow and, if positive, expand
   attribution controls.
```

The cleanest next implementation slice is a transition-balanced DataLoader mode
for flow/action-head training plus a dataset expansion plan.
