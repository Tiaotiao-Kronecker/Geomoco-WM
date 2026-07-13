# Gate 3.2i Boundary-Index Localizer

## Purpose

Gate 3.2h proved that transition-local gripper correction can beat the
Gate 3.1f/Gate 3.1g reference if `close_step/open_step` timing is known, but
predicted sparse boundary masks failed to recover that oracle gain.

Gate 3.2i is the short-budget follow-up: replace sparse per-step boundary CE
with a window-level close/open step-index localizer.

```text
Predict close index and open index separately as [B, 2, H+1].
Class H is no-event.
Apply existing close/open step residuals at the predicted indices.
```

This keeps the residual mechanism fixed and changes only the localization
objective/readout.

## Code Changes

Implemented the boundary-index head and reporting path:

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
scripts/audit_predicted_event_mixture_action_head_groups.py
tests/test_motion_prior_action_head.py
tests/test_predicted_event_mixture_action_head_group_audit.py
```

New config:

```text
gripper_boundary_index_mode=boundary_index
gripper_boundary_index_loss_weight
```

New metrics:

```text
boundary_index_pred_*
gripper_boundary_index_ce
gripper_boundary_index_accuracy
gripper_boundary_index_close_accuracy
gripper_boundary_index_open_accuracy
gripper_boundary_index_close_within1
gripper_boundary_index_open_within1
```

## Commands

Smoke:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2i_boundary_index_smoke_seed7 \
  --event-top-m 4 \
  --num-samples 8 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
  --gripper-step-target-mode boundary_start \
  --gripper-step-residual-blend positive_only \
  --gripper-step-residual-loss-weight 0.0 \
  --gripper-step-loss-weight 0.0 \
  --gripper-boundary-index-mode boundary_index \
  --gripper-boundary-index-loss-weight 0.1 \
  --selection-metric boundary_index_pred_mse \
  --max-windows 128 \
  --epochs 1 \
  --batch-size 16 \
  --seed 7 \
  --device cpu \
  --quiet
```

Full short-budget run:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --output-dir outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --gripper-step-residual-mode event_step \
  --gripper-step-target-mode boundary_start \
  --gripper-step-residual-blend positive_only \
  --gripper-step-residual-loss-weight 0.0 \
  --gripper-step-loss-weight 0.0 \
  --gripper-boundary-index-mode boundary_index \
  --gripper-boundary-index-loss-weight 0.1 \
  --selection-metric boundary_index_pred_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Evaluation and group audit:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --num-samples 16 \
  --batch-size 64 \
  --device cuda \
  --seed ${seed}

.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed${seed}/group_audit_3pass.json \
  --num-eval-passes 3 \
  --num-samples 16 \
  --batch-size 64 \
  --device cuda \
  --seed ${seed}
```

## Results

Mean over seeds 7 and 17, repeated eval:

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.2h oracle boundary mask | 0.032018 | 0.133072 | 0.021841 | 0.116469 |
| Gate 3.2h best predicted mask | 0.035201 | 0.155339 | 0.021902 | 0.145723 |
| Gate 3.2i base | 0.034878 | 0.151459 | 0.022817 | 0.134422 |
| Gate 3.2i boundary-index readout | 0.035053 | 0.152685 | 0.022868 | 0.135682 |

Per-seed repeated eval:

| seed | base MSE | boundary-index MSE | base transition MSE | boundary-index transition MSE |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.037113 | 0.037221 | 0.146008 | 0.146373 |
| 17 | 0.032643 | 0.032885 | 0.122835 | 0.124992 |

Group audit mean, transition/sustain groups:

| group | base MSE | boundary-index MSE | base gripper MSE | boundary-index gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| sustain | 0.022805 | 0.022855 | 0.069764 | 0.070112 |
| transition | 0.134393 | 0.135641 | 0.826043 | 0.834775 |

Boundary-index localization, mean over seeds:

| metric | value |
| --- | ---: |
| overall close/open accuracy, including no-event | 0.940462 |
| event fraction | 0.054302 |
| close exact accuracy on event windows | 0.069527 |
| open exact accuracy on event windows | 0.023539 |
| close within-1 on event windows | 0.249839 |
| open within-1 on event windows | 0.160918 |

## Interpretation

Gate 3.2i does not recover the Gate 3.2h oracle-boundary gain.

The apparent seed-17 improvement is not a boundary-localizer success: the base
actions from that checkpoint already reach `0.032643`, while the
boundary-index readout worsens them to `0.032885`. The same within-checkpoint
pattern holds for seed 7.

The localizer still mostly learns the no-event prior. Its aggregate accuracy is
high only because no-event dominates; exact close/open localization remains
weak, especially for open events.

Compared with Gate 3.2h predicted masks, the boundary-index objective is a
small improvement in transition MSE, but not a material one:

```text
Gate 3.2h best predicted-mask overall MSE      = 0.035201
Gate 3.2i boundary-index overall MSE           = 0.035053
Gate 3.1f/Gate 3.1g deployable reference       = 0.034767
Gate 3.2h oracle-boundary upper bound          = 0.032018
```

It does not beat the deployable reference and it does not approach the oracle
mask.

## Decision

Do not promote Gate 3.2i.

Treat the short-budget option A as exhausted. The next mainline should pivot to
a richer temporal/flow action decoder or another action trajectory model that
directly models gripper transition shape, instead of another deterministic
boundary-routing head.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable
reference.

## Verification

Checks:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head tests.test_predicted_event_mixture_action_head_group_audit tests.test_gripper_boundary_timing_audit
.venv/bin/python -m compileall -q src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_group_audit.py
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_group_audit.py
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_2i_boundary_index_smoke_seed7/
outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_2i_boundary_index_top4_k16_seed17/
```
