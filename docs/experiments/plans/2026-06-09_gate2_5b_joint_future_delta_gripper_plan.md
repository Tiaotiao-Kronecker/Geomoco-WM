# Gate 2.5b-Joint Future-Delta-Gripper Predictor Plan

- Date: 2026-06-09
- Status: completed
- Position: after the modular predicted EEF + predicted gripper bridge showed
  useful visual attribution but compounded separate predictor errors.

## Purpose

Train one deterministic predictor for the joint target:

```text
future_delta_ee + future_gripper
```

instead of composing two separately trained heads. The goal is to make the
future representation match the `future_delta_gripper` action decoder input and
avoid modular error compounding.

## Hypothesis

The first `lambda_action=0.030` joint run may underweight the action loss
because the flat joint motion target is dominated by gripper command scale.
Therefore test:

```text
lambda_action = 0.030
lambda_action = 0.300
```

If `0.300` improves action MSE, treat the problem as loss-scale mismatch rather
than a failure of the joint representation.

## Dataset And Inputs

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

Run branches:

```text
task/proprio, lambda=0.300
visual patchpool, lambda=0.030
visual patchpool, lambda=0.300
shuffled visual patchpool, lambda=0.300
```

## Metrics

Promotion metric:

```text
downstream action MSE through future_delta_gripper action decoder
```

Supporting metrics:

```text
SE(3) action MSE
action gripper MSE
future EEF MSE
future gripper MSE
transition event accuracy / macro-F1 / transition step within 1
```

## Decision Rule

Promote the joint output space if the real visual joint predictor:

1. beats modular predicted EEF + predicted gripper;
2. beats the previous best EEF-only learned prior;
3. beats task/proprio and shuffled visual controls;
4. keeps event metrics positive relative to controls.

