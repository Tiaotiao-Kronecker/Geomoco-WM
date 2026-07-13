# Gate 2.4f Flat Vs Structured Oracle Ranks

- Date: 2026-06-08
- Status: completed
- Scope: decide whether Gate 2.4e structured scorers should be promoted after
  adding `SE(3)` and gripper-aware oracle/rank diagnostics.

## Main Table

| scorer target | action MSE | flat rank | SE(3) rank | SE(3)+gripper rank | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| flat action MSE | 0.040190 | 6.624765 | 7.768547 | 7.451381 | keep as deployable baseline |
| `SE(3)` | 0.040642 | 7.563506 | 7.289575 | 7.663167 | diagnostic only |
| `SE(3)+gripper` | 0.040441 | 7.220221 | 7.408181 | 7.364062 | diagnostic only |

Lower rank is better. `1.0` would mean the scorer always selected the oracle
best sample from K=16 candidates.

## What The New Oracle Metrics Tell Us

The structured scorer targets are not useless. They move the rank in the
expected direction:

- `SE(3)` target improves `SE(3)` oracle rank from `7.768547` to `7.289575`;
- `SE(3)+gripper` target improves `SE(3)+gripper` oracle rank from `7.451381`
  to `7.364062`.

But this is not enough to improve the current downstream action-value metric:

```text
flat action-MSE scorer: 0.040190
SE(3) scorer:           0.040642
SE(3)+gripper scorer:   0.040441
```

## Mainline Implication

The bottleneck is not just the scalar metric used as the ranking target.

The scorer is missing information that distinguishes a physically plausible
future-motion sample from a sample that will decode into useful, executable
control. The next branch should therefore add hard negatives or explicit
executability proxies instead of continuing to hand-tune `SE(3)`/gripper
weights.

## Next Gate

Gate 2.4g should test hard-negative / executability-aware readout:

1. construct hard negatives from samples that look close under `SE(3)` but have
   worse decoded-action error;
2. add an auxiliary pairwise loss so the scorer learns to prefer samples that
   are both geometrically plausible and action-useful;
3. keep flat action MSE as the promotion metric;
4. report `SE(3)` and `SE(3)+gripper` oracle ranks as diagnostics, not as the
   sole promotion criterion.
