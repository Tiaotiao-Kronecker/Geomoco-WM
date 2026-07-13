# Gate 2.5a Visual Future-Gripper/Event Predictor Plan

- Date: 2026-06-09
- Status: completed
- Position: after Gate 2.4i proved that EEF-only future motion misses the
  gripper/event channel.

## Purpose

Gate 2.4i showed a large oracle gap:

```text
GT future EEF only          -> action MSE 0.031474, gripper MSE 0.184683
GT future EEF + GT gripper  -> action MSE 0.004202, gripper MSE 0.000241
```

Gate 2.5a asks whether the missing gripper/event channel can be predicted from
deployable current context:

```text
RGB/DINO + proprio + suite/task -> future_gripper
```

This is not yet a full deployable policy input because the bridge diagnostic
still uses GT future EEF. Its role is to test whether visual grounding can
recover the gripper/open-close timing that EEF-only motion lacks.

## Dataset

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

## Variants

Use three variants:

```text
task_proprio
visual_patchpool
shuffled_visual_patchpool
```

`visual_patchpool` is the real two-camera DINO patch-pooled cross-attention
branch. `shuffled_visual_patchpool` keeps visual statistics but destroys
window alignment, so it is the visual attribution control.

## Metrics

Report three layers:

1. Future-gripper regression:

```text
gripper MSE / MAE
```

2. Event fidelity:

```text
event type accuracy
macro-F1
transition type accuracy
transition step within 1
```

3. Action-bridge value:

```text
GT future EEF + predicted future gripper -> future_delta_gripper action decoder
```

This reports flat action metrics, SE(3) metrics, and gripper action metrics.

## Promotion Rule

Promote the branch if real visual features beat both task/proprio and shuffled
visual controls on gripper regression, transition-event metrics, and downstream
action bridge metrics.

If the bridge improves over Gate 2.4i EEF-only oracle, the missing event
channel has been partially repaired by a deployable visual predictor.

## Next If Positive

Move to Gate 2.5b:

```text
predicted future EEF + predicted future gripper
```

Then move to Gate 2.5c:

```text
GeoMoCo-cVAE future_delta_ee + future_gripper/event output
```

