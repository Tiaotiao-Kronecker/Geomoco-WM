# Gate 3.2g Boundary Quality Audit And Positive-Only Repair

## Purpose

Gate 3.2f showed that boundary-start supervision gives a small transition gain,
but raw step accuracy is misleading because positive boundary labels are only
about `1.36%` of all horizon steps.

Gate 3.2g tests two narrower questions:

```text
1. Does the boundary-start head actually detect close/open boundary steps?
2. Can a transition-local residual avoid perturbing no-boundary/sustain steps?
```

## Code Changes

Boundary audit:

```text
scripts/audit_gripper_boundary_timing_head.py
tests/test_gripper_boundary_timing_audit.py
```

The audit reports:

```text
positive AP
argmax precision/recall/F1
threshold precision/recall/F1
close/open step top-1 exact and within-1 localization
```

Action-head repair:

```text
gripper_step_residual_blend = all_classes | positive_only
```

The default remains `all_classes` for backward compatibility. The new
`positive_only` mode ignores class-0 (`no_boundary`) residuals and blends only
`close_start/open_start` residuals:

```text
residual_t = p(close_start_t) * r(close_start_t)
           + p(open_start_t)  * r(open_start_t)
```

Additional calibration test:

```text
gripper_step_positive_loss_weight = 20.0
```

This upweights nonzero boundary labels in the CE term.

## Commands

Boundary audit for Gate 3.2f:

```bash
.venv/bin/python scripts/audit_gripper_boundary_timing_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed}/boundary_audit_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

Positive-only training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2g_boundary_positive_only_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
  --gripper-step-target-mode boundary_start \
  --gripper-step-residual-blend positive_only \
  --gripper-step-residual-loss-weight 1.0 \
  --gripper-step-loss-weight 0.1 \
  --selection-metric step_routed_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Positive-class-weighted training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2g_boundary_posw20_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
  --gripper-step-target-mode boundary_start \
  --gripper-step-residual-blend positive_only \
  --gripper-step-residual-loss-weight 1.0 \
  --gripper-step-loss-weight 0.1 \
  --gripper-step-positive-loss-weight 20.0 \
  --selection-metric step_routed_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Each branch was evaluated with:

```text
repeated_eval_5pass.json
group_stress_3pass.json
boundary_audit_3pass.json
```

## Results

Mean over seeds 7 and 17, repeated eval:

| branch | readout | overall MSE | gripper MSE | sustain MSE | transition MSE | boundary AP | argmax recall | argmax precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 3.2f all-classes | base | 0.035453 | 0.154943 | 0.023101 | 0.137941 | - | - | - |
| Gate 3.2f all-classes | step | 0.035605 | 0.156003 | 0.023363 | 0.137172 | 0.096503 | 0.007233 | 0.254444 |
| Gate 3.2g positive-only | base | 0.034938 | 0.153773 | 0.022594 | 0.137499 | - | - | - |
| Gate 3.2g positive-only | step | 0.035204 | 0.155630 | 0.022986 | 0.136703 | 0.090235 | 0.001634 | 0.015176 |
| Gate 3.2g posw20 | base | 0.035649 | 0.153919 | 0.022703 | 0.143033 | - | - | - |
| Gate 3.2g posw20 | step | 0.035778 | 0.154816 | 0.022957 | 0.142132 | 0.095948 | 0.532779 | 0.094722 |

Mean group audit:

| branch | group | base MSE | step MSE | base gripper MSE | step gripper MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Gate 3.2f | sustain | 0.023105 | 0.023369 | 0.071115 | 0.072961 |
| Gate 3.2f | transition | 0.137955 | 0.137198 | 0.851940 | 0.846643 |
| Gate 3.2g positive-only | sustain | 0.022594 | 0.022985 | 0.071545 | 0.074281 |
| Gate 3.2g positive-only | transition | 0.137499 | 0.136702 | 0.838882 | 0.833302 |
| Gate 3.2g posw20 | sustain | 0.022712 | 0.022967 | 0.065844 | 0.067629 |
| Gate 3.2g posw20 | transition | 0.143160 | 0.142263 | 0.886781 | 0.880499 |

Boundary audit highlights:

| branch | positive AP | argmax recall | argmax precision | pred positive fraction | close top1 | open top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 3.2f all-classes | 0.096503 | 0.007233 | 0.254444 | 0.000558 | 0.213999 | 0.148681 |
| Gate 3.2g positive-only | 0.090235 | 0.001634 | 0.015176 | 0.000841 | 0.199420 | 0.139365 |
| Gate 3.2g posw20 | 0.095948 | 0.532779 | 0.094722 | 0.076041 | 0.222889 | 0.152954 |

## Interpretation

The boundary audit explains the Gate 3.2f failure:

```text
boundary accuracy is high because almost every step is no_boundary.
argmax recall is about 0.7%, so the head almost never fires on close/open starts.
```

Positive-only residual blending is a weak mechanism-positive result:

```text
transition MSE: 0.137172 -> 0.136703
transition gripper MSE: 0.846643 -> 0.833302
```

but it is still not deployable:

```text
overall MSE: 0.034767 Gate 3.1f/g reference
overall MSE: 0.035204 Gate 3.2g positive-only step
```

Positive class weighting changes the classifier behavior, but not in a useful
way. It raises argmax recall to `0.532779`, but precision is only `0.094722`
and transition MSE gets much worse:

```text
positive-only step transition MSE = 0.136703
posw20 step transition MSE        = 0.142132
```

So the next bottleneck is not merely class imbalance. The model can be forced
to emit boundary positives, but the positives are not precise enough to drive
action residuals.

## Verification

Checks run:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head tests.test_gripper_boundary_timing_audit tests.test_predicted_event_mixture_action_head_group_audit
.venv/bin/python -m compileall -q src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_gripper_boundary_timing_head.py tests/test_motion_prior_action_head.py tests/test_gripper_boundary_timing_audit.py
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_gripper_boundary_timing_head.py tests/test_motion_prior_action_head.py tests/test_gripper_boundary_timing_audit.py
```

Execution:

```text
CPU smoke completed for positive_only and posw20.
Full GPU train/eval/audit completed for seeds 7 and 17.
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_2g_boundary_positive_only_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_2g_boundary_positive_only_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_2g_boundary_posw20_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_2g_boundary_posw20_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_2g_boundary_positive_only_smoke_seed7/
outputs/motion_prior_action_head/gate3_2g_boundary_posw20_smoke_seed7/
```

## Decision

Do not promote Gate 3.2g.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable
reference:

```text
action MSE = 0.034767
gripper MSE = 0.150052
```

Gate 3.2g confirms that deterministic residual repairs are now hitting a
precision/localization wall. The next deterministic action-head branch should
not be another CE-weighted boundary classifier. Either use a direct
transition-local gripper trajectory objective with oracle boundary masks, or
move to a richer temporal action head that models gripper changes as a
sequence-level event rather than a sparse per-step class.
