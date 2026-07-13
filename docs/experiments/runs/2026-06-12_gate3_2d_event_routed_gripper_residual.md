# Gate 3.2d Event-Routed Gripper Residual

## Purpose

Gate 3.2a showed that the current best event-aware top-4 action-head interface
fails mainly on open/close transition windows. Gate 3.2b showed that scalar
transition weighting is trainable but hurts overall and sustain windows. Gate
3.2c showed that a parallel gripper regression head is not enough.

Gate 3.2d tests the next minimal first-principles hypothesis:

```text
keep one shared SE(3) action head
+ add an event-family route
+ add family-specific gripper residuals
+ only modify the gripper action channel through the routed residual
```

The goal is not to replace the full action head. The goal is to test whether
explicit event routing can repair gripper transition timing without damaging
sustain behavior.

## Model Change

`MotionPriorActionHead` now supports:

```text
gripper_residual_mode = event_family
gripper_route_families = sustain, transition_close, transition_open
```

Forward behavior remains backward-compatible:

```text
model(...) -> base actions
```

The routed branch is exposed through:

```text
model.forward_with_aux(...) -> {
  actions: [B,H,7],
  routed_actions: [B,H,7],
  gripper_route_logits: [B,3],
  gripper_route_probs: [B,3],
  gripper_residuals: [B,H,3]
}
```

`routed_actions` copies the base action and adds a blended residual only to the
last gripper channel.

## Training Objective

Base action loss remains:

```text
L_base = mean((a_base - a_gt)^2)
```

Routed action loss:

```text
L_routed = mean((a_routed - a_gt)^2)
```

Route supervision:

```text
L_route = CE(route_logits, event_family)
```

Total:

```text
L = L_base + lambda_routed * L_routed + lambda_route * L_route
```

Configuration:

```text
event_top_m = 4
num_samples = 16
sample_feature_mode = event_rank_prob
gripper_residual_mode = event_family
gripper_residual_loss_weight = 1.0
gripper_route_loss_weight = 0.1
selection_metric = routed_mse
seeds = 7, 17
```

## Commands

Training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2d_event_routed_gripper_residual_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-residual-mode event_family \
  --gripper-residual-loss-weight 1.0 \
  --gripper-route-loss-weight 0.1 \
  --selection-metric routed_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Repeated eval:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2d_event_routed_gripper_residual_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2d_event_routed_gripper_residual_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

Group audit:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2d_event_routed_gripper_residual_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2d_event_routed_gripper_residual_top4_k16_seed${seed}/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Results

Mean over seeds 7 and 17, repeated eval:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE | route accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.035624 | 0.154709 | 0.023131 | 0.138891 | 0.921017 |
| routed | 0.035774 | 0.155760 | 0.023355 | 0.138409 | 0.921017 |

Mean over seeds 7 and 17, group audit:

| group | base MSE | routed MSE | base gripper MSE | routed gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| sustain | 0.023134 | 0.023358 | 0.069629 | 0.071195 |
| transition | 0.138891 | 0.138406 | 0.859218 | 0.855824 |
| transition_open | 0.159891 | 0.159651 | 0.961180 | 0.959496 |
| transition_close | 0.118982 | 0.118223 | 0.761398 | 0.756080 |

## Interpretation

Gate 3.2d is mechanism-positive but deployable-negative.

Positive:

```text
The event-family route is learnable: route accuracy is about 0.92.
The routed residual slightly improves transition MSE and transition gripper MSE.
```

Negative:

```text
The improvement is small.
Sustain windows get worse.
Overall action MSE and overall gripper MSE get worse.
```

So the current window-level event-routed residual is not enough to promote. It
knows the event family, but it still lacks step-level timing/execution structure
inside the action chunk.

## Decision

Do not promote Gate 3.2d as the default action head.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable reference:

```text
action MSE = 0.034767
gripper MSE = 0.150052
```

Next mainline should move from window-level event family routing to step-level
event timing:

```text
Gate 3.2e: step-wise gripper/event-timing head or temporal transition mask.
```

The key question becomes:

```text
Can the action head infer when inside the horizon to open/close, not merely
whether the whole window is a sustain/open-transition/close-transition window?
```

