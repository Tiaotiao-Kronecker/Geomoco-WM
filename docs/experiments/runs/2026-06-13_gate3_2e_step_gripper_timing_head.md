# Gate 3.2e Step-Wise Gripper Command-Timing Head

## Purpose

Gate 3.2d showed that window-level event-family routing is too coarse. The
event family is readable, but a single window-level gripper residual does not
fix open/close timing.

Gate 3.2e tests the next first-principles question:

```text
Can the action head use step-wise gripper command-state supervision to improve
open/close transition windows?
```

The branch keeps the Gate 3.1f/Gate 3.1g predicted top-4 event-mixture
interface and adds a per-step gripper head:

```text
base action head
+ per-step gripper command logits: sustain / close / open
+ per-step gripper residuals blended by the logits
```

## Model Change

`MotionPriorActionHead` now supports:

```text
gripper_step_residual_mode = event_step
gripper_step_classes = sustain, close, open
```

The new opt-in outputs are:

```text
step_routed_actions: [B,H,7]
gripper_step_logits: [B,H,3]
gripper_step_probs: [B,H,3]
gripper_step_residuals: [B,H,3]
```

`step_routed_actions` only changes the gripper action channel.

## Training Objective

Base loss:

```text
L_base = mean((a_base - a_gt)^2)
```

Step-routed action loss:

```text
L_step = mean((a_step_routed - a_gt)^2)
```

Per-step command-state loss:

```text
L_step_ce = CE(step_logits, command_state)
```

where command state is derived from the ground-truth gripper command:

```text
close if gripper >= 0.5
open  if gripper <= -0.5
sustain otherwise
```

Total:

```text
L = L_base + lambda_step * L_step + lambda_step_ce * L_step_ce
```

Configuration:

```text
event_top_m = 4
num_samples = 16
sample_feature_mode = event_rank_prob
gripper_step_residual_mode = event_step
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
  --output-dir outputs/motion_prior_action_head/gate3_2e_step_gripper_timing_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
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
  --checkpoint outputs/motion_prior_action_head/gate3_2e_step_gripper_timing_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2e_step_gripper_timing_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

Group audit:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2e_step_gripper_timing_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2e_step_gripper_timing_top4_k16_seed${seed}/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Results

Mean over seeds 7 and 17, repeated eval:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE | step accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.035254 | 0.153039 | 0.022961 | 0.137397 | 0.947207 |
| step-routed | 0.035397 | 0.154042 | 0.023084 | 0.137707 | 0.947207 |

Mean over seeds 7 and 17, group audit:

| group | base MSE | step-routed MSE | base gripper MSE | step-routed gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| sustain | 0.022965 | 0.023087 | 0.069584 | 0.070435 |
| transition | 0.137415 | 0.137737 | 0.847309 | 0.849564 |
| transition_open | 0.147680 | 0.147155 | 0.872512 | 0.868832 |
| transition_close | 0.127658 | 0.128735 | 0.822536 | 0.830080 |

## Interpretation

Gate 3.2e is a useful negative ablation.

The step command classifier is easy:

```text
step accuracy = 0.947207
```

But this does not improve action prediction:

```text
overall MSE: 0.035254 -> 0.035397
transition MSE: 0.137397 -> 0.137707
gripper MSE: 0.153039 -> 0.154042
```

The reason is visible in the label definition. LIBERO gripper commands are
mostly explicit open/close commands at every step, so command-state prediction
is not the same as transition-boundary timing. The hard part is not:

```text
is this step commanded open or close?
```

The hard part is:

```text
which step starts the close/open transition relative to the previous gripper
state?
```

## Decision

Do not promote Gate 3.2e.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable
reference:

```text
action MSE = 0.034767
gripper MSE = 0.150052
```

Next mainline:

```text
Gate 3.2f: step-wise transition-boundary timing.
```

Instead of supervising every step's command state, Gate 3.2f should use
`close_step` and `open_step` from the event-mode audit JSON to supervise:

```text
no_boundary / close_start / open_start
```

This matches the actual bottleneck identified by Gate 3.2a: open/close
transition timing.

