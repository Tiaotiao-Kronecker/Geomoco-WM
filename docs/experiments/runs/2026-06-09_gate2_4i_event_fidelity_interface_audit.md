# Gate 2.4i Event-Fidelity Interface Audit

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.4i
- Purpose: test whether EEF-only future motion lacks the gripper/open-close
  phase information needed for action decoding.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| windows | 16,518 |
| horizon | 8 |
| split policy | episode |
| seeds | 7, 17 |
| device | cuda for training, cpu for event eval |

## Code

```text
src/geomoco_wm/data/window_dataset.py
scripts/train_oracle_action_decoder.py
scripts/evaluate_action_decoder_events.py
tests/test_libero_hdf5_export.py
```

New oracle motion modes:

```text
future_gripper
future_delta_gripper
```

## Commands

Example:

```bash
.venv/bin/python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7 \
  --motion-mode future_delta_gripper \
  --split-by episode \
  --seed 7 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --quiet
```

Event evaluation:

```bash
.venv/bin/python scripts/evaluate_action_decoder_events.py \
  --checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/action_decoder_event_eval/gate2_4i_future_delta_gripper_seed7.json \
  --device cpu \
  --quiet
```

## Artifacts

Training outputs:

```text
outputs/oracle_action_decoder/gate2_4i_none_seed7/
outputs/oracle_action_decoder/gate2_4i_none_seed17/
outputs/oracle_action_decoder/gate2_4i_future_gripper_seed7/
outputs/oracle_action_decoder/gate2_4i_future_gripper_seed17/
outputs/oracle_action_decoder/gate2_4i_future_delta_seed7/
outputs/oracle_action_decoder/gate2_4i_future_delta_seed17/
outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/
outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed17/
```

Event evaluation:

```text
outputs/action_decoder_event_eval/gate2_4i_<motion_mode>_seed<seed>.json
```

## Mean Results

Mean over seeds 7 and 17:

| input | action MSE | action MAE | SE(3) MSE | trans L2 | rot deg | gripper MSE | gripper MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context only | 0.066010 | 0.147124 | 0.034921 | 0.380476 | 2.233651 | 0.252545 | 0.353775 |
| future gripper only | 0.020666 | 0.085811 | 0.023969 | 0.317188 | 2.130801 | 0.000848 | 0.022678 |
| future EEF only | 0.031474 | 0.079508 | 0.005939 | 0.149313 | 1.048033 | 0.184683 | 0.280764 |
| future EEF + gripper | 0.004202 | 0.036480 | 0.004863 | 0.129674 | 0.961951 | 0.000241 | 0.011763 |

Event fidelity:

| input | event acc | macro-F1 | transition acc | step within 1 |
| --- | ---: | ---: | ---: | ---: |
| context only | 0.718071 | 0.442005 | 0.203855 | 0.055590 |
| future gripper only | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| future EEF only | 0.784005 | 0.456044 | 0.178190 | 0.054159 |
| future EEF + gripper | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Relative reduction from EEF-only to EEF+gripper:

| metric | reduction |
| --- | ---: |
| action MSE | 86.65% |
| action MAE | 54.12% |
| gripper MSE | 99.87% |
| gripper MAE | 95.81% |
| SE(3) MSE | 18.13% |
| translation L2 | 13.15% |
| rotation deg | 8.21% |

## Interpretation

This is the strongest evidence so far that the current EEF-only interface is
missing the gripper/event channel.

Key points:

- GT future EEF-only motion is excellent for SE(3) action components, but poor
  for gripper action and transition timing.
- GT future gripper alone almost perfectly recovers close/open transition
  labels, because the labels are derived from the gripper command sequence.
- GT future EEF + gripper gives a large action-MSE gain over EEF-only and
  removes nearly all gripper error.

This validates the user's intuition: the previous readout/cVAE bottleneck is
basically a gripper/event information bottleneck, not just a scorer weakness.

## Decision

Do not continue tuning EEF-only sample readout as the main branch.

Promote a new world-motion target:

```text
future_delta_ee + future_gripper/event
```

Next mainline:

```text
Gate 2.5a: visual future-gripper/event predictor
Gate 2.5b: visual future-EEF+gripper predictor
Gate 2.5c: GeoMoCo-cVAE with EEF+gripper/event output
```

The first deployable step should predict future gripper/event channels from
visual/proprio/task context, then pass the predicted EEF+gripper representation
through the action decoder.

## Caveat

`future_gripper` and `future_delta_gripper` are oracle upper bounds. They use
future action gripper commands and are not deployable by themselves. Their role
is to prove that the missing channel matters.

