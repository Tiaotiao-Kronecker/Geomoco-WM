# Gate 3.5a Small Residual Flow Decoder

## Purpose

Gate 3.4 was a small positive for a controlled joint temporal action decoder.
Gate 3.4b showed that the full-aligned checkpoint uses K-sample diversity at
runtime, while the trained `mean_repeated` control remains close. Gate 3.4c
showed that explicit motion-regret sample scoring does not improve action
value.

Gate 3.5a tests the next smallest richer decoder:

```text
Keep the Gate 3.4 temporal action decoder.
Add a small residual rectified-flow branch after temporal_actions.
Do not replace the full action policy.
Keep the predicted top-4 event/rank/prob interface fixed.
```

## Implementation

Added:

```text
flow_action_decoder_mode=rectified_mlp
flow_action_loss_weight
flow_matching_loss_weight
```

Model outputs:

```text
actions
temporal_actions
flow_actions
flow_action_velocity
flow_action_residual
```

Training target:

```text
residual_target = ground_truth_actions - temporal_actions
z ~ N(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) * z + t * residual_target
velocity_target = residual_target - z
```

Deployable eval uses the deterministic one-step residual:

```text
flow_actions = temporal_actions + velocity_theta(x_t=0, t=0, cond)
```

The branch is intentionally a residual adapter, not a full replacement policy.

## Commands

Full-aligned short-budget runs:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
    --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
    --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
    --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
    --output-dir outputs/motion_prior_action_head/gate3_5a_flow_top4_k16_seed${seed} \
    --event-top-m 4 \
    --num-samples 16 \
    --sample-feature-mode event_rank_prob \
    --temporal-action-decoder-mode sequence_mlp \
    --temporal-action-loss-weight 1.0 \
    --flow-action-decoder-mode rectified_mlp \
    --flow-action-loss-weight 1.0 \
    --flow-matching-loss-weight 0.1 \
    --selection-metric flow_action_mse \
    --epochs 20 \
    --batch-size 64 \
    --seed ${seed} \
    --device cuda \
    --quiet
done
```

Repeated eval:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
    --checkpoint outputs/motion_prior_action_head/gate3_5a_flow_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_5a_flow_top4_k16_seed${seed}/repeated_eval.json \
    --num-eval-passes 5 \
    --device cuda
done
```

## Results

5-pass repeated eval:

| seed | temporal MSE | flow MSE | temporal transition | flow transition | temporal gripper | flow gripper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.038590 | 0.038607 | 0.148318 | 0.149636 | 0.175264 | 0.174984 |
| 17 | 0.032195 | 0.032635 | 0.124474 | 0.126023 | 0.136461 | 0.137875 |
| mean | 0.035392 | 0.035621 | 0.136396 | 0.137830 | 0.155863 | 0.156430 |

Gate 3.4 reference, mean over seeds 7 and 17:

```text
temporal_action_mse:              0.034262
temporal_action_transition_mse:   0.131311
temporal_action_gripper_mse:      0.149383
temporal_action_sustain_mse:      0.022542
```

Gate 3.5a does not pass the full-aligned promotion check:

```text
Gate 3.5a flow MSE:       0.035621
Gate 3.4 temporal MSE:    0.034262
```

It also fails the same-checkpoint decoder-gain test:

```text
Gate 3.5a temporal MSE:   0.035392
Gate 3.5a flow MSE:       0.035621
decoder gain:             -0.000229
```

Transition also regresses:

```text
Gate 3.5a temporal transition: 0.136396
Gate 3.5a flow transition:    0.137830
Gate 3.4 temporal transition: 0.131311
```

## Interpretation

This is a short-budget negative result.

The smallest residual flow adapter does not improve the Gate 3.4 temporal
decoder. Within the same checkpoint, the flow readout is worse than
`temporal_actions`; relative to Gate 3.4, both the temporal branch and the flow
branch are worse. This suggests the current flow objective/branch interferes
with the shared representation or overfits residual noise rather than learning
a useful action residual trajectory.

This does not show that every flow/diffusion action decoder is bad. It shows
that the minimal one-step rectified residual adapter is not the right next
promotion branch under the current short-budget setup.

## Decision

Do not run full attribution controls for Gate 3.5a. The first full-aligned
promotion check failed, so controls would only explain a non-promoted branch.

Keep the implementation and audit support as reusable plumbing:

```text
flow_action_* metrics
flow readout support in usage audit
```

Next branch should either:

```text
1. decouple residual flow training from the shared temporal decoder, e.g. freeze
   a Gate 3.4 checkpoint and train a post-hoc residual adapter;
2. reduce the flow branch to a deterministic residual MLP without flow matching
   to test whether the noise objective is the harmful piece;
3. move upstream to improve transition/event candidate quality before adding
   another richer decoder.
```

The most attribution-clean immediate follow-up is option 1: freeze the promoted
Gate 3.4 checkpoint and train a separate post-hoc residual adapter. That would
test residual action modeling without degrading the temporal decoder itself.

## Verification

Checks:

```text
.venv/bin/python -m compileall -q src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_usage.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_usage_audit.py
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_usage.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_usage_audit.py
.venv/bin/python -m pytest tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_usage_audit.py
```

Smoke:

```text
outputs/motion_prior_action_head/gate3_5a_flow_smoke_seed7/
outputs/motion_prior_action_head/gate3_5a_flow_smoke_seed7/repeated_eval_smoke.json
outputs/motion_prior_action_head/gate3_5a_flow_smoke_seed7/usage_audit_smoke.json
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_5a_flow_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_5a_flow_top4_k16_seed17/
docs/experiments/plans/2026-06-16_gate3_5a_small_residual_flow_decoder_plan.md
```
