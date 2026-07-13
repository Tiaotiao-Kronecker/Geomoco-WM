# Gate 3.3 Minimal Temporal Gripper Trajectory Decoder

## Purpose

Gate 3.2h showed that oracle `close_step/open_step` masks make
transition-local gripper correction valuable, but Gate 3.2i showed that a
short-budget deployable boundary-index localizer still cannot recover that
oracle gain.

Gate 3.3 tests the smallest richer decoder pivot:

```text
Keep the Gate 3.1f/g predicted top-4 event/rank/prob sample interface fixed.
Add only a temporal gripper residual sequence branch.
Do not replace the full action policy.
```

## Code Changes

Implemented an optional gripper trajectory residual branch:

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
scripts/audit_predicted_event_mixture_action_head_groups.py
tests/test_motion_prior_action_head.py
```

New config:

```text
gripper_trajectory_residual_mode=temporal_mlp
gripper_trajectory_residual_loss_weight
```

New model outputs:

```text
trajectory_routed_actions
gripper_trajectory_residuals
```

New metrics:

```text
trajectory_routed_*
gripper_trajectory_residual_mse
```

## Commands

CPU plumbing smoke:

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

Short-budget run:

```bash
for seed in 7 17; do
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
done
```

Repeated eval and group audit:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
    --checkpoint outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed${seed}/repeated_eval.json \
    --num-eval-passes 5 \
    --device cuda

  .venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
    --checkpoint outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed${seed}/group_audit.json \
    --num-eval-passes 3 \
    --device cuda
done
```

## Results

Mean over seeds 7 and 17, repeated eval:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.3 base output | 0.035516 | 0.153991 | 0.023151 | 0.137769 |
| Gate 3.3 trajectory-routed output | 0.035655 | 0.154966 | 0.023274 | 0.138046 |

Per-seed repeated eval:

| seed | base MSE | trajectory-routed MSE | base transition MSE | trajectory-routed transition MSE |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.038139 | 0.038292 | 0.147050 | 0.147223 |
| 17 | 0.032894 | 0.033019 | 0.128488 | 0.128868 |

The branch is consistently worse than the same checkpoint's base action output.

## Interpretation

Gate 3.3 validates the plumbing for a minimal temporal gripper residual
sequence, but the short-budget model does not improve action quality.

This is not an attribution-positive result. The full aligned GeoMoCo-WM
sample interface is still present, but the new trajectory branch fails the
first required comparison:

```text
trajectory-routed output should improve over the same decoder's base output.
```

It does not. The branch slightly worsens overall, gripper, sustain, and
transition MSE on both seeds.

Because the main aligned branch is negative, the full control matrix is not
worth expanding yet. Controls would answer whether a positive result comes from
the motion prior or from decoder capacity; here there is no positive result to
attribute.

## Decision

Do not promote Gate 3.3 `temporal_mlp`.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable
reference.

The next branch should not be another shallow additive gripper residual on top
of the same base action head. Viable next directions are:

```text
1. a fuller temporal action-sequence decoder that jointly decodes gripper and motion;
2. a small flow/diffusion-style action residual decoder with strict controls;
3. richer contact/object/event supervision before increasing decoder capacity.
```

## Verification

Checks:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head
.venv/bin/python -m unittest tests.test_predicted_event_mixture_action_head_group_audit tests.test_gripper_boundary_timing_audit
.venv/bin/python -m compileall -q src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_group_audit.py
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_group_audit.py
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_3_gripper_traj_smoke_seed7/
outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_3_gripper_traj_top4_k16_seed17/
```
