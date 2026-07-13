# Gate 2.5b Predicted EEF + Predicted Gripper Bridge Plan

- Date: 2026-06-09
- Status: completed
- Position: after Gate 2.5a showed that visual future-gripper prediction
  partially repairs the EEF-only oracle interface.

## Purpose

Gate 2.5a still used a privileged input:

```text
GT future EEF + predicted future gripper
```

Gate 2.5b removes that privilege in the fastest diagnostic way:

```text
predicted future EEF + predicted future gripper
  -> future_delta_gripper action decoder
```

This tests whether separately trained future EEF and future gripper predictors
compose into a useful joint action interface.

## Branches

Use existing checkpoints:

```text
task/proprio:
  EEF:     gate2_1_suite_task
  gripper: gate2_5a_task_proprio

visual patchpool:
  EEF:     gate2_3b_action_aware_lam003_patchpool4_crossattn
  gripper: gate2_5a_visual_patchpool

shuffled visual patchpool:
  EEF:     gate2_2c_shuffled_patchpool4_crossattn
  gripper: gate2_5a_shuffled_visual_patchpool
```

The branch is intentionally modular. It should not be treated as the final
joint predictor if errors compound.

## Metrics

Report:

```text
predicted EEF future-motion metrics
predicted gripper metrics
joint action MSE / MAE
joint SE(3) action metrics
joint gripper action metrics
```

Also run two diagnostic decompositions for the visual branch:

```text
GT EEF + predicted gripper
predicted EEF + GT gripper
predicted EEF + predicted gripper
```

## Decision Rule

If modular `predicted EEF + predicted gripper` beats the best EEF-only learned
prior, it can be promoted as the deterministic joint interface.

If it beats controls but does not beat the EEF-only learned prior, treat it as
a positive visual-attribution diagnostic and train a joint
`future_delta_gripper` predictor instead of hard-composing separately trained
heads.

