# Gate 3.2f Step-Wise Transition-Boundary Timing Head

## Purpose

Gate 3.2e showed that per-step gripper command-state supervision is too easy
and does not repair open/close transition windows. LIBERO often emits explicit
open/close commands across many steps, so command sign is not the same as the
start of a gripper state transition.

Gate 3.2f tests the sharper timing hypothesis:

```text
Can the action head use close_step/open_step boundary-start supervision to
improve open/close transition windows?
```

This branch keeps the Gate 3.1f/Gate 3.1g predicted top-4 event-mixture
interface and reuses the Gate 3.2e step-routed gripper residual head, but
switches the per-step target from command state to transition boundary start.

## Model Change

The existing step-routed gripper branch is reused:

```text
base action head
+ per-step boundary logits
+ per-step gripper residuals blended by boundary probabilities
```

The branch only changes the gripper action channel in the step-routed output:

```text
step_routed_actions[..., -1] =
    base_actions[..., -1] + blended_step_boundary_residual
```

New training/eval configuration:

```text
gripper_step_target_mode = boundary_start
gripper_step_classes = no_boundary, close_start, open_start
```

The default remains:

```text
gripper_step_target_mode = command_state
```

so older Gate 3.2e checkpoints and configs remain compatible.

## Training Objective

Base action loss:

```text
L_base = mean((a_base - a_gt)^2)
```

Step-routed action loss:

```text
L_step = mean((a_step_routed - a_gt)^2)
```

Boundary-start classification loss:

```text
L_boundary = CE(step_logits, boundary_start_target)
```

Boundary targets are built from the event-mode audit records:

```text
target[t] = close_start if close_step == t
target[t] = open_start  if open_step == t
target[t] = no_boundary otherwise
```

Total:

```text
L = L_base + lambda_step * L_step + lambda_boundary * L_boundary
```

Configuration:

```text
event_top_m = 4
num_samples = 16
sample_feature_mode = event_rank_prob
gripper_step_residual_mode = event_step
gripper_step_target_mode = boundary_start
gripper_step_residual_loss_weight = 1.0
gripper_step_loss_weight = 0.1
selection_metric = step_routed_mse
seeds = 7, 17
```

## Commands

Training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
  --gripper-step-target-mode boundary_start \
  --gripper-step-residual-loss-weight 1.0 \
  --gripper-step-loss-weight 0.1 \
  --selection-metric step_routed_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Repeated eval:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

Group audit:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed${seed}/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Results

Mean over seeds 7 and 17, repeated eval:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE | boundary acc. | boundary frac. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.035453 | 0.154943 | 0.023101 | 0.137941 | 0.986058 | 0.013576 |
| step-routed | 0.035605 | 0.156003 | 0.023363 | 0.137172 | 0.986058 | 0.013576 |

The target is extremely sparse. A no-boundary majority classifier would reach
approximately:

```text
1 - boundary_fraction = 0.986424
```

so boundary accuracy alone is not evidence that the classifier solved boundary
localization.

Mean over seeds 7 and 17, group audit:

| group | base MSE | step-routed MSE | base gripper MSE | step-routed gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| sustain | 0.023105 | 0.023369 | 0.071115 | 0.072961 |
| transition | 0.137955 | 0.137198 | 0.851940 | 0.846643 |
| transition_open | 0.150525 | 0.149920 | 0.893725 | 0.889488 |
| transition_close | 0.126053 | 0.125132 | 0.811938 | 0.805494 |

Per-seed repeated eval:

| seed | readout | overall MSE | gripper MSE | sustain MSE | transition MSE | boundary acc. | boundary frac. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | base | 0.037910 | 0.169775 | 0.024576 | 0.141830 | 0.984946 | 0.014391 |
| 7 | step-routed | 0.038123 | 0.171269 | 0.024912 | 0.141093 | 0.984946 | 0.014391 |
| 17 | base | 0.032997 | 0.140110 | 0.021625 | 0.134053 | 0.987169 | 0.012761 |
| 17 | step-routed | 0.033086 | 0.140737 | 0.021814 | 0.133251 | 0.987169 | 0.012761 |

## Interpretation

Gate 3.2f is mechanism-positive but not deployable.

Positive:

```text
transition MSE improves: 0.137941 -> 0.137172
transition gripper MSE improves: 0.851940 -> 0.846643
```

Negative:

```text
overall MSE worsens: 0.035453 -> 0.035605
gripper MSE worsens: 0.154943 -> 0.156003
sustain MSE worsens: 0.023101 -> 0.023363
```

The branch improves exactly the targeted transition subgroup inside its own
run, but it still does not beat the Gate 3.1f/Gate 3.1g deployable reference:

```text
Gate 3.1f/Gate 3.1g reference overall MSE = 0.034767
Gate 3.2f step-routed overall MSE          = 0.035605
```

The high boundary-step accuracy should be interpreted cautiously because the
boundary labels are sparse. The useful signal is the small transition MSE gain,
not the raw accuracy number.

## Verification

Implemented and checked:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head
.venv/bin/python -m unittest tests.test_imports tests.test_predicted_event_mixture_action_head_group_audit
.venv/bin/python -m compileall -q scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py tests/test_motion_prior_action_head.py
.venv/bin/ruff check scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py tests/test_motion_prior_action_head.py
```

Additional execution:

```text
dry-run passed for the full Gate 3.2f seed 7 command
CPU smoke trained and evaluated under gate3_2f_boundary_timing_smoke_seed7
full GPU training/eval completed for seeds 7 and 17
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_2f_boundary_timing_smoke_seed7/
```

## Decision

Do not promote Gate 3.2f as the default action head.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable
reference:

```text
action MSE = 0.034767
gripper MSE = 0.150052
```

Gate 3.2b through Gate 3.2f jointly show that transition timing is real and
trainable, but simple deterministic repairs only move error between transition
and sustain regimes. The next deterministic slice should measure boundary
precision/recall explicitly and use a transition-local objective or calibrated
boundary gate before escalating to a flow/diffusion action head.
