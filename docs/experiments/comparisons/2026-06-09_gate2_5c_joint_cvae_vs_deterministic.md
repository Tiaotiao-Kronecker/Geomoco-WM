# Gate 2.5c Joint cVAE vs Deterministic Joint Baseline

- Date: 2026-06-09
- Scope: joint `future_delta_ee + future_gripper/event` cVAE.

## Main Table

| branch | action MSE | SE(3) MSE | gripper MSE | transition acc |
| --- | ---: | ---: | ---: | ---: |
| deterministic joint | 0.040688 | 0.020486 | 0.161903 | 0.560270 |
| cVAE prior mean | 0.043816 | 0.020473 | 0.183879 | 0.571094 |
| cVAE best-of-K action | 0.022139 | 0.017020 | 0.052857 | not yet measured |
| shuffled cVAE prior mean | 0.068816 | 0.035716 | 0.267414 | 0.223375 |

## What Passed

The cVAE sample set is valuable:

```text
prior mean action MSE: 0.043816
best-of-K action MSE: 0.022139
```

Real visual grounding matters:

```text
real prior action MSE:     0.043816
shuffled prior action MSE: 0.068816
```

Event fidelity is also aligned:

```text
real transition acc:     0.571094
shuffled transition acc: 0.223375
```

## What Did Not Pass

The raw prior mean is not yet a deployable improvement over deterministic
joint:

```text
deterministic joint: 0.040688
cVAE prior mean:    0.043816
```

Random sample mean is worse than prior mean, so sampling without readout is not
usable.

## Next Decision

Move to:

```text
Gate 2.5d joint cVAE sample readout/scorer
```

The readout should try to recover part of the best-of-K coverage while keeping
event fidelity and visual controls in the table.

