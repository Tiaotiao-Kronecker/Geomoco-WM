# Gate 2.4i Event-Fidelity Interface Audit Plan

- Date: 2026-06-09
- Status: completed
- Position: after Gate 2.4h-d negative event-aware readout diagnostic.

## Purpose

Gate 2.4h showed that close/open transition timing is the weak point. Gate
2.4i asks whether the current interface is missing information:

```text
future EEF-only motion -> frozen / learned action decoder -> gripper action
```

The suspected issue is that EEF `SE(3)` motion alone cannot uniquely determine
the gripper command phase.

## Oracle Inputs

Use the same `ActionDecoder` and same train/val split, but change the oracle
motion representation:

```text
none                  # direct context baseline
future_gripper        # GT future gripper commands only
future_delta          # GT future EEF deltas only
future_delta_gripper  # GT future EEF deltas + GT future gripper commands
```

`future_gripper` and `future_delta_gripper` are oracle upper bounds. They are
not deployable, because they use future action gripper commands.

## Metrics

- flat action MSE / MAE;
- split SE(3) and gripper metrics;
- transition event accuracy / macro-F1;
- transition type accuracy;
- transition step within 1.

## Decision Rule

If `future_delta_gripper` greatly improves over `future_delta`, the current
GeoMoCo-WM world-motion interface should be expanded beyond EEF-only motion.

The next deployable branch should predict future gripper/event channels from
visual/proprio/task context, rather than merely changing the ScoreNet readout.

