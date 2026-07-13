# Gate 3.3 Minimal Temporal Gripper Trajectory Decoder Plan

## Purpose

Gate 3.2h showed that oracle `close_step/open_step` timing makes
transition-local gripper correction valuable. Gate 3.2i then tested a
short-budget deployable boundary-index localizer, but the predicted index
readout still failed to beat the Gate 3.1f/Gate 3.1g reference.

Gate 3.3 pivots from point boundary prediction to trajectory modeling:

```text
Model the gripper transition trajectory directly, while keeping the
Gate 3.1f/Gate 3.1g GeoMoCo-WM sample interface fixed.
```

The goal is not to replace the whole action policy with a larger black box.
The first implementation should add only a minimal temporal gripper
trajectory/residual branch.

## Diagnosis Being Tested

The current failure is not that event information is absent. The failure is
that close/open action chunks require a phaseful gripper trajectory, not a
single brittle predicted boundary point.

Evidence:

```text
Gate 3.2a: transition windows dominate remaining error.
Gate 3.2h: oracle boundary masks improve overall MSE to 0.032018.
Gate 3.2f/g/h/i: predicted boundary/localizer variants do not recover that gain.
```

## Minimal Model Change

Keep the existing `MotionPriorActionHead` sample aggregation and base action
prediction.

Add one optional gripper trajectory branch:

```text
features -> temporal gripper residual sequence [B, H]
base actions[..., -1] + residual sequence -> trajectory-routed actions
```

First version:

```text
mode: gripper_trajectory_residual
loss: action loss on trajectory-routed actions, optionally transition-weighted
selection metric: trajectory_routed_mse or trajectory_routed_transition_mse
```

This is deliberately smaller than a full diffusion/flow policy. It directly
tests whether a temporal gripper sequence can repair transition shape without
changing the GeoMoCo-WM prior.

## Attribution Controls

| control | purpose |
| --- | --- |
| full event/rank/prob samples | main candidate; tests the complete Gate 3.1f/Gate 3.1g interface under the new decoder |
| shuffled event metadata | checks whether aligned event identity still matters |
| rank/prob-only metadata | separates event confidence/order from event identity |
| mean replacement | checks whether sample-set diversity is still used |
| context-only/no-prior decoder | checks whether gains come from the stronger decoder alone |
| same decoder capacity without motion-prior samples | controls for parameter count and temporal branch capacity |

The key comparison is always under the same decoder family:

```text
full aligned GeoMoCo-WM samples > context-only/no-prior
full aligned GeoMoCo-WM samples > shuffled metadata
full aligned GeoMoCo-WM samples > mean replacement
```

If those do not hold, the decoder may have swallowed the attribution.

## First Executable Slice

Start with the smallest implementation and smoke:

```text
input interface: predicted event-mixture top-4, K=16, event_rank_prob
seeds: smoke seed 7 first
training: max_windows=128, epochs=1, CPU
then full short run: seeds 7 and 17, epochs=20, GPU
```

Only after the main branch is positive should the full control matrix run.

## Metrics

Primary:

```text
trajectory_routed_mse
trajectory_routed_gripper_mse
trajectory_routed_transition_mse
trajectory_routed_sustain_mse
```

Secondary:

```text
base mse/gripper/transition/sustain
transition_group group audit
per-event-family group audit
physical translation/rotation metrics from action_metrics
```

The expected positive pattern is:

```text
transition gripper MSE improves materially
transition MSE improves
sustain MSE does not regress enough to erase the gain
overall MSE beats 0.034767
```

## Promotion Criteria

Promote Gate 3.3 only if the full aligned branch:

```text
beats Gate 3.1f/Gate 3.1g overall MSE 0.034767;
improves transition MSE over 0.134087;
does not achieve the gain equally with context-only/no-prior controls;
does not achieve the gain equally with shuffled event metadata;
preserves or improves gripper MSE relative to 0.150052.
```

If it only improves transition MSE while worsening overall/sustain, archive it
as mechanism-positive but not deployable, matching the Gate 3.2b standard.

## Initial Commands

Smoke target:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_3_gripper_traj_smoke_seed7 \
  --event-top-m 4 \
  --num-samples 8 \
  --sample-feature-mode event_rank_prob \
  --gripper-trajectory-residual-mode temporal_mlp \
  --gripper-trajectory-residual-loss-weight 1.0 \
  --selection-metric trajectory_routed_mse \
  --max-windows 128 \
  --epochs 1 \
  --batch-size 16 \
  --seed 7 \
  --device cpu \
  --quiet
```

Full short-budget target:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-trajectory-residual-mode temporal_mlp \
  --gripper-trajectory-residual-loss-weight 1.0 \
  --selection-metric trajectory_routed_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

## Stop Conditions

Stop this branch if:

```text
smoke shows the trajectory readout is identical to base because the residual
does not train;
trajectory-routed overall MSE is worse than base on both seeds;
controls show context-only/no-prior matches the full branch;
the improvement comes only from sustain/geometry rather than transition gripper.
```

## Expected Interpretation

A positive Gate 3.3 result would say:

```text
GeoMoCo-WM's event-aware sample interface remains useful, but the downstream
action decoder needs a temporal gripper trajectory branch to consume close/open
transition information.
```

A negative result would say:

```text
The bottleneck is not only boundary localization or shallow gripper trajectory
decoding; the project should consider a fuller action-sequence decoder,
diffusion/flow action model, or richer contact/object supervision.
```
