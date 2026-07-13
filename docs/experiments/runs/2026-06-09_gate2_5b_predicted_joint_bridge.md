# Gate 2.5b Predicted EEF + Predicted Gripper Bridge

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.5b modular bridge diagnostic
- Purpose: test whether separately trained predicted EEF and predicted gripper
  futures compose into a useful `future_delta_gripper` action interface.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |
| split policy | episode |
| seeds | 7, 17 |
| eval device | cpu |

## Code

```text
scripts/evaluate_predicted_joint_action_bridge.py
scripts/train_future_motion_predictor.py
```

The new bridge script supports:

```text
--eef-source predicted|gt
--gripper-source predicted|gt
```

This allows decomposition of EEF prediction error and gripper prediction error
inside the same joint action decoder.

## Commands

Main visual branch example:

```bash
.venv/bin/python scripts/evaluate_predicted_joint_action_bridge.py \
  --eef-predictor-checkpoint outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed7/model.pt \
  --gripper-predictor-checkpoint outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed7/model.pt \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --output-json outputs/predicted_joint_action_bridge/gate2_5b_visual_patchpool_seed7.json \
  --seed 7 \
  --device cpu \
  --quiet
```

Diagnostic upper-bound example:

```bash
.venv/bin/python scripts/evaluate_predicted_joint_action_bridge.py \
  --eef-predictor-checkpoint outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed7/model.pt \
  --gripper-predictor-checkpoint outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed7/model.pt \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --output-json outputs/predicted_joint_action_bridge/gate2_5b_visual_patchpool_pred_eef_gt_gripper_seed7.json \
  --seed 7 \
  --device cpu \
  --gripper-source gt \
  --quiet
```

## Artifacts

```text
outputs/predicted_joint_action_bridge/gate2_5b_task_proprio_seed7.json
outputs/predicted_joint_action_bridge/gate2_5b_task_proprio_seed17.json
outputs/predicted_joint_action_bridge/gate2_5b_visual_patchpool_seed7.json
outputs/predicted_joint_action_bridge/gate2_5b_visual_patchpool_seed17.json
outputs/predicted_joint_action_bridge/gate2_5b_shuffled_visual_patchpool_seed7.json
outputs/predicted_joint_action_bridge/gate2_5b_shuffled_visual_patchpool_seed17.json
outputs/predicted_joint_action_bridge/gate2_5b_visual_patchpool_pred_eef_gt_gripper_seed7.json
outputs/predicted_joint_action_bridge/gate2_5b_visual_patchpool_pred_eef_gt_gripper_seed17.json
```

## Main Results

Mean over seeds 7 and 17:

| branch | action MSE | action MAE | SE(3) MSE | trans MAE (m) | rot deg | action gripper MSE | action gripper MAE | EEF MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 0.079633 | 0.164476 | 0.039495 | 0.010546 | 2.151134 | 0.320460 | 0.410378 | 0.000929 | 0.324415 |
| visual patchpool | 0.050333 | 0.115699 | 0.029825 | 0.008960 | 2.200837 | 0.173377 | 0.161379 | 0.000782 | 0.172088 |
| shuffled visual patchpool | 0.065466 | 0.133920 | 0.038015 | 0.010132 | 2.155526 | 0.230170 | 0.221882 | 0.000985 | 0.233726 |

Real visual remains the best branch:

| comparison | action MSE reduction | action gripper MSE reduction | EEF MSE reduction | gripper MSE reduction |
| --- | ---: | ---: | ---: | ---: |
| visual vs task/proprio | 36.79% | 45.90% | 15.81% | 46.95% |
| visual vs shuffled visual | 23.12% | 24.67% | 20.67% | 26.37% |

## Error Decomposition

Mean over seeds 7 and 17:

| input | action MSE | action MAE | SE(3) MSE | gripper MSE | gripper MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT EEF + predicted visual gripper | 0.028987 | 0.058115 | 0.004943 | 0.173254 | 0.161812 |
| predicted visual EEF + GT gripper | 0.025443 | 0.094384 | 0.029632 | 0.000312 | 0.013272 |
| predicted visual EEF + predicted visual gripper | 0.050333 | 0.115699 | 0.029825 | 0.173377 | 0.161379 |
| GT EEF only | 0.031474 | 0.079508 | 0.005939 | 0.184683 | 0.280764 |
| GT EEF + GT gripper | 0.004202 | 0.036480 | 0.004863 | 0.000241 | 0.011763 |

## Interpretation

Gate 2.5b modular bridge is a positive visual-attribution diagnostic but is not
promoted as the deterministic joint interface.

Key points:

- Real visual beats task/proprio and shuffled controls in the full predicted
  joint bridge.
- However, modular `predicted EEF + predicted gripper` action MSE `0.050333`
  is worse than the previous best EEF-only learned prior action MSE `0.042090`.
- `predicted EEF + GT gripper` has low flat action MSE because the gripper term
  is nearly solved, but its SE(3) MSE remains poor (`0.029632`). This shows EEF
  prediction noise is still a serious bottleneck in the joint decoder.
- `GT EEF + predicted gripper` keeps SE(3) strong and improves over EEF-only,
  which confirms Gate 2.5a's gripper result. The two predicted branches
  together compound errors.

## Decision

Do not hard-compose separately trained EEF and gripper heads as the final
method.

Next mainline:

```text
Gate 2.5b-joint: train a deterministic future_delta_gripper predictor
```

The joint predictor should learn the EEF/gripper target together and use the
`future_delta_gripper` action decoder during action-aware training. If that is
positive, promote the same output space to GeoMoCo-cVAE in Gate 2.5c.

## Verification

```text
.venv/bin/python -m compileall scripts/evaluate_predicted_joint_action_bridge.py
.venv/bin/ruff check scripts/evaluate_predicted_joint_action_bridge.py
```

Both passed. The target script smoke with `--max-windows 512` also passed.

