# Gate 3.2h Oracle-Boundary Transition-Local Gripper Correction

## Purpose

Gate 3.2g showed that sparse boundary classification was the bottleneck:
positive-only residuals helped transition gripper MSE slightly, but predicted
boundary positives were too rare or too noisy.

Gate 3.2h asks a sharper upper-bound question:

```text
If the true close_step/open_step boundary mask is known, can a local gripper
residual beat the Gate 3.1f/Gate 3.1g event-aware action-head reference?
```

If yes, the next question is whether predicted boundary masks can recover the
oracle-mask gain.

## Code Changes

Implemented oracle and predicted boundary-mask residual readouts:

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
scripts/audit_predicted_event_mixture_action_head_groups.py
tests/test_motion_prior_action_head.py
tests/test_predicted_event_mixture_action_head_group_audit.py
```

New training/eval config:

```text
gripper_step_oracle_boundary_residual_loss_weight
```

New readout metrics:

```text
oracle_step_routed_*
pred_boundary_argmax_*
pred_boundary_t0p05_*
pred_boundary_t0p10_*
pred_boundary_t0p20_*
pred_boundary_t0p30_*
pred_boundary_t0p50_*
```

`oracle_step_routed_*` applies the predicted residual only at oracle positive
`close_step/open_step` targets. `pred_boundary_*` applies the same residuals
using the model's own boundary logits.

## Commands

Training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
  --gripper-step-target-mode boundary_start \
  --gripper-step-residual-blend positive_only \
  --gripper-step-residual-loss-weight 0.0 \
  --gripper-step-loss-weight 0.1 \
  --gripper-step-oracle-boundary-residual-loss-weight 1.0 \
  --selection-metric oracle_step_routed_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Evaluation and audits:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda

.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed}/group_stress_predmask_3pass.json \
  --num-eval-passes 3 \
  --device cuda

.venv/bin/python scripts/audit_gripper_boundary_timing_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed${seed}/boundary_audit_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Results

Mean over seeds 7 and 17, repeated eval:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.2h base | 0.035145 | 0.154961 | 0.021841 | 0.145678 |
| Gate 3.2h soft step-routed | 0.034989 | 0.153872 | 0.022187 | 0.141261 |
| Gate 3.2h oracle boundary mask | 0.032018 | 0.133072 | 0.021841 | 0.116469 |

Mean predicted-mask group audit:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| base | 0.035164 | 0.155080 | 0.021853 | 0.145762 |
| soft step-routed | 0.035000 | 0.153936 | 0.022198 | 0.141278 |
| oracle boundary mask | 0.032035 | 0.133180 | 0.021853 | 0.116536 |
| predicted argmax | 0.035201 | 0.155339 | 0.021902 | 0.145723 |
| predicted threshold 0.05 | 0.044260 | 0.218753 | 0.029007 | 0.170924 |
| predicted threshold 0.10 | 0.039628 | 0.186327 | 0.025839 | 0.154139 |
| predicted threshold 0.20 | 0.036666 | 0.165593 | 0.023379 | 0.147111 |
| predicted threshold 0.30 | 0.035632 | 0.158357 | 0.022358 | 0.145985 |
| predicted threshold 0.50 | 0.035201 | 0.155339 | 0.021902 | 0.145723 |

Boundary-quality audit, mean over seeds:

| metric | value |
| --- | ---: |
| positive AP | 0.098873 |
| argmax recall | 0.012026 |
| argmax precision | 0.226716 |
| argmax predicted-positive fraction | 0.000723 |
| close top-1 exact | 0.222833 |
| open top-1 exact | 0.142825 |

## Interpretation

Gate 3.2h is oracle-positive:

```text
Gate 3.1f/g reference overall MSE = 0.034767
Gate 3.2h oracle boundary MSE     = 0.032018
Gate 3.2h oracle transition MSE   = 0.116469
```

The true boundary mask cleanly protects sustain windows and applies the
residual exactly where the gripper timing error is concentrated.

But Gate 3.2h is deployable-negative:

```text
best predicted-mask overall MSE is about 0.035201
best predicted-mask transition MSE is about 0.145723
```

The predicted mask does not recover the oracle-mask gain. Low thresholds fire
too often and damage sustain/overall quality; argmax or high thresholds fire
too rarely and are effectively close to the base model.

## Decision

Do not promote Gate 3.2h as a deployable action head.

Do promote the diagnosis: local gripper correction is valuable if boundary
timing is known. The next mainline should not add another sparse CE variant.
It should either:

```text
1. improve boundary localization with a different temporal objective/head; or
2. move to a richer temporal/flow action decoder that models the gripper
   transition trajectory directly.
```

## Verification

Checks:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head tests.test_predicted_event_mixture_action_head_group_audit tests.test_gripper_boundary_timing_audit
.venv/bin/python -m compileall -q src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_group_audit.py
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_group_audit.py
```

Execution:

```text
CPU smoke completed:
outputs/motion_prior_action_head/gate3_2h_oracle_boundary_smoke_seed7/

Full GPU train/eval/audit completed for seeds 7 and 17:
outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed17/
```
