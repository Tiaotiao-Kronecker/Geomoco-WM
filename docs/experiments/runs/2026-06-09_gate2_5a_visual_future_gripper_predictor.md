# Gate 2.5a Visual Future-Gripper/Event Predictor

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.5a
- Purpose: predict the future gripper/event channel from visual/proprio/task
  context, then test whether it repairs the EEF-only action bridge gap exposed
  in Gate 2.4i.

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
| event/bridge eval device | cpu |

## Code

```text
scripts/train_future_motion_predictor.py
scripts/evaluate_future_gripper_events.py
scripts/evaluate_predicted_gripper_action_bridge.py
src/geomoco_wm/data/window_dataset.py
src/geomoco_wm/data/event_labels.py
tests/test_future_motion_predictor.py
```

New target mode:

```text
--motion-mode future_gripper
```

The bridge diagnostic uses:

```text
GT future EEF + predicted future gripper -> future_delta_gripper action decoder
```

## Commands

Training example:

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed7 \
  --motion-mode future_gripper \
  --condition-on suite_task \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --epochs 20 \
  --batch-size 256 \
  --device cuda \
  --seed 7 \
  --quiet
```

Event evaluation example:

```bash
.venv/bin/python scripts/evaluate_future_gripper_events.py \
  --checkpoint outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed7/model.pt \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/future_gripper_event_eval/gate2_5a_visual_patchpool_seed7.json \
  --device cpu \
  --quiet
```

Action bridge example:

```bash
.venv/bin/python scripts/evaluate_predicted_gripper_action_bridge.py \
  --gripper-predictor-checkpoint outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed7/model.pt \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --output-json outputs/predicted_gripper_action_bridge/gate2_5a_visual_patchpool_seed7.json \
  --device cpu \
  --quiet
```

## Artifacts

Predictors:

```text
outputs/future_gripper_predictor/gate2_5a_task_proprio_seed7/
outputs/future_gripper_predictor/gate2_5a_task_proprio_seed17/
outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed7/
outputs/future_gripper_predictor/gate2_5a_visual_patchpool_seed17/
outputs/future_gripper_predictor/gate2_5a_shuffled_visual_patchpool_seed7/
outputs/future_gripper_predictor/gate2_5a_shuffled_visual_patchpool_seed17/
```

Event evaluation:

```text
outputs/future_gripper_event_eval/gate2_5a_<variant>_seed<seed>.json
```

Action bridge evaluation:

```text
outputs/predicted_gripper_action_bridge/gate2_5a_<variant>_seed<seed>.json
```

## Future-Gripper Regression

Mean over seeds 7 and 17:

| variant | gripper MSE | gripper MAE |
| --- | ---: | ---: |
| task/proprio | 0.324415 | 0.412950 |
| visual patchpool | 0.172088 | 0.158186 |
| shuffled visual patchpool | 0.233726 | 0.225239 |

Relative to task/proprio, real visual reduces gripper MSE by `46.95%` and
gripper MAE by `61.69%`.

Relative to shuffled visual, real visual reduces gripper MSE by `26.37%` and
gripper MAE by `29.77%`.

## Event Fidelity

Mean over seeds 7 and 17:

| variant | event acc | macro-F1 | transition acc | step within 1 |
| --- | ---: | ---: | ---: | ---: |
| task/proprio | 0.595893 | 0.311213 | 0.014386 | 0.003003 |
| visual patchpool | 0.888486 | 0.647211 | 0.634542 | 0.270724 |
| shuffled visual patchpool | 0.818344 | 0.544600 | 0.481249 | 0.174244 |

Real visual features strongly improve close/open transition prediction. The
shuffled branch remains above task/proprio, so there is a dataset/task/scene
prior, but the real branch is still clearly better than shuffled visual.

## Action Bridge

Mean over seeds 7 and 17:

| variant | action MSE | action MAE | SE(3) MSE | trans MAE (m) | rot deg | gripper MSE | gripper MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 0.050046 | 0.094904 | 0.005244 | 0.003421 | 0.976799 | 0.318862 | 0.408828 |
| visual patchpool | 0.028987 | 0.058115 | 0.004943 | 0.003259 | 0.962564 | 0.173254 | 0.161812 |
| shuffled visual patchpool | 0.037060 | 0.066858 | 0.005031 | 0.003287 | 0.965983 | 0.229232 | 0.221129 |

Relative reductions for real visual:

| comparison | action MSE | gripper MSE | gripper MAE |
| --- | ---: | ---: | ---: |
| visual vs task/proprio | 42.08% | 45.66% | 60.42% |
| visual vs shuffled visual | 21.78% | 24.42% | 26.82% |

## Relation To Gate 2.4i Bounds

| input | action MSE | gripper MSE |
| --- | ---: | ---: |
| Gate 2.4i future EEF only | 0.031474 | 0.184683 |
| Gate 2.5a GT EEF + predicted visual gripper | 0.028987 | 0.173254 |
| Gate 2.4i future EEF + GT gripper | 0.004202 | 0.000241 |

Gate 2.5a improves over the EEF-only oracle interface, so the visual
future-gripper branch repairs part of the missing channel. It remains far from
the GT future-gripper upper bound, so the remaining gap is still large.

## Interpretation

Gate 2.5a is a positive mechanism result.

Main points:

- Visual grounding predicts future gripper commands much better than
  task/proprio metadata.
- Real visual features beat shuffled visual features, so the gain is not only a
  dataset prior.
- The predicted gripper channel improves the action bridge over EEF-only,
  confirming that the earlier bottleneck was a real interface issue.
- The bridge diagnostic still uses GT future EEF, so it is not yet deployable.

## Decision

Promote the target representation from:

```text
future_delta_ee
```

to:

```text
future_delta_ee + future_gripper/event
```

Next mainline:

```text
Gate 2.5b: predicted future EEF + predicted future gripper
Gate 2.5c: GeoMoCo-cVAE with EEF+gripper/event output
```

Gate 2.5b should determine whether the benefit survives when both EEF and
gripper futures are predicted from current visual/proprio/task context.

## Verification

```text
.venv/bin/python -m compileall scripts/evaluate_future_gripper_events.py scripts/evaluate_predicted_gripper_action_bridge.py scripts/train_future_motion_predictor.py tests/test_future_motion_predictor.py
.venv/bin/ruff check scripts/evaluate_future_gripper_events.py scripts/evaluate_predicted_gripper_action_bridge.py scripts/train_future_motion_predictor.py tests/test_future_motion_predictor.py
.venv/bin/python -m unittest tests.test_future_motion_predictor tests.test_libero_hdf5_export
```

All passed before the bridge evaluation.

