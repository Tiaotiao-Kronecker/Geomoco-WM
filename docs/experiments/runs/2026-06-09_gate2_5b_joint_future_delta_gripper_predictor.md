# Gate 2.5b-Joint Future-Delta-Gripper Predictor

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.5b-joint
- Purpose: train a deterministic visual predictor whose output is directly
  `future_delta_ee + future_gripper`, then test whether it beats the modular
  bridge and the previous EEF-only learned prior.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
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
| training device | cuda |
| event eval device | cpu |

## Code

```text
scripts/train_future_motion_predictor.py
scripts/evaluate_future_gripper_events.py
tests/test_future_motion_predictor.py
```

The event evaluator now supports both:

```text
future_gripper
future_delta_gripper
```

For `future_delta_gripper`, it extracts the final `H` gripper dimensions from
the flattened `[6H + H]` prediction.

## Commands

Real visual joint predictor, seed 7:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/future_motion_predictor/gate2_5b_joint_action_aware_lam03_patchpool4_crossattn_seed7 \
  --motion-mode future_delta_gripper \
  --condition-on suite_task \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --action-aware-loss-weight 0.3 \
  --epochs 20 \
  --batch-size 256 \
  --device cuda \
  --seed 7 \
  --quiet
```

Event evaluation:

```bash
.venv/bin/python scripts/evaluate_future_gripper_events.py \
  --checkpoint outputs/future_motion_predictor/gate2_5b_joint_action_aware_lam03_patchpool4_crossattn_seed7/model.pt \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/future_gripper_event_eval/gate2_5b_joint_visual_patchpool_lam03_seed7.json \
  --device cpu \
  --quiet
```

## Artifacts

Lambda sweep:

```text
outputs/future_motion_predictor/gate2_5b_joint_action_aware_lam003_patchpool4_crossattn_seed7/
outputs/future_motion_predictor/gate2_5b_joint_action_aware_lam003_patchpool4_crossattn_seed17/
outputs/future_motion_predictor/gate2_5b_joint_action_aware_lam03_patchpool4_crossattn_seed7/
outputs/future_motion_predictor/gate2_5b_joint_action_aware_lam03_patchpool4_crossattn_seed17/
```

Controls:

```text
outputs/future_motion_predictor/gate2_5b_joint_task_proprio_lam03_seed7/
outputs/future_motion_predictor/gate2_5b_joint_task_proprio_lam03_seed17/
outputs/future_motion_predictor/gate2_5b_joint_shuffled_visual_lam03_seed7/
outputs/future_motion_predictor/gate2_5b_joint_shuffled_visual_lam03_seed17/
```

Event evaluation:

```text
outputs/future_gripper_event_eval/gate2_5b_joint_<variant>_lam03_seed<seed>.json
```

## Lambda Result

Mean over seeds 7 and 17:

| branch | action MSE | action MAE | SE(3) MSE | trans MAE (m) | rot deg | gripper MSE | gripper MAE | future flat MSE | future EEF MSE | future gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual lambda 0.030 | 0.049103 | 0.115536 | 0.028630 | 0.009028 | 2.146322 | 0.171939 | 0.158063 | 0.025367 | 0.000902 | 0.172160 |
| visual lambda 0.300 | 0.040688 | 0.101703 | 0.020486 | 0.007407 | 2.073053 | 0.161903 | 0.162981 | 0.023847 | 0.000946 | 0.161253 |

Increasing the action-aware weight from `0.030` to `0.300` improves action MSE
by `17.14%`. This supports the loss-scale interpretation.

## Visual Controls

Mean over seeds 7 and 17, all with `lambda_action=0.300`:

| branch | action MSE | action MAE | SE(3) MSE | trans MAE (m) | rot deg | gripper MSE | gripper MAE | future flat MSE | future EEF MSE | future gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 0.084648 | 0.170563 | 0.043964 | 0.011114 | 2.494170 | 0.328752 | 0.402574 | 0.048472 | 0.001331 | 0.331320 |
| visual patchpool | 0.040688 | 0.101703 | 0.020486 | 0.007407 | 2.073053 | 0.161903 | 0.162981 | 0.023847 | 0.000946 | 0.161253 |
| shuffled visual | 0.063790 | 0.136704 | 0.038224 | 0.010213 | 2.421466 | 0.217181 | 0.222541 | 0.032282 | 0.001407 | 0.217531 |

Relative reductions for real visual:

| comparison | action MSE | SE(3) MSE | gripper MSE |
| --- | ---: | ---: | ---: |
| visual vs task/proprio | 51.93% | 53.40% | 50.75% |
| visual vs shuffled visual | 36.22% | 46.41% | 25.45% |

## Event Fidelity

Mean over seeds 7 and 17:

| branch | event acc | macro-F1 | transition acc | step within 1 |
| --- | ---: | ---: | ---: | ---: |
| task/proprio | 0.611115 | 0.316683 | 0.009672 | 0.001502 |
| visual patchpool | 0.878580 | 0.612640 | 0.560270 | 0.234025 |
| shuffled visual | 0.824836 | 0.514325 | 0.330051 | 0.120085 |

Real visual remains clearly better than both controls on transition fidelity.

## Relation To Prior Baselines

| reference | action MSE |
| --- | ---: |
| modular predicted EEF + predicted gripper | 0.050333 |
| Gate 2.5b-joint visual lambda 0.030 | 0.049103 |
| Gate 2.5b-joint visual lambda 0.300 | 0.040688 |
| previous best EEF-only learned prior | 0.042090 |
| GT EEF + predicted visual gripper | 0.028987 |
| GT EEF + GT gripper | 0.004202 |

The joint predictor with stronger action-aware loss is the first learned
EEF+gripper branch to beat the previous EEF-only learned prior.

## Interpretation

Gate 2.5b-joint is positive.

Key points:

- Direct joint training beats hard-composing separate EEF and gripper heads.
- A stronger action-aware weight is necessary because the joint motion loss has
  different scale from EEF-only motion loss.
- Real visual grounding beats task/proprio and shuffled visual controls on
  action metrics, SE(3), gripper metrics, and transition-event fidelity.
- The gain over the best EEF-only prior is modest (`0.042090 -> 0.040688`), so
  this is a promotion of the representation direction, not a final method.

## Decision

Promote the output space:

```text
future_delta_ee + future_gripper/event
```

Next mainline:

```text
Gate 2.5c: GeoMoCo-cVAE with joint EEF+gripper/event output
```

Use `lambda_action=0.300` as the deterministic joint baseline for cVAE
comparison. Keep event metrics and shuffled visual controls mandatory.

## Verification

```text
.venv/bin/python -m compileall scripts/evaluate_future_gripper_events.py
.venv/bin/ruff check scripts/evaluate_future_gripper_events.py
```

Both passed before event evaluation.

