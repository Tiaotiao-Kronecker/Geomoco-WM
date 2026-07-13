# Gate 2.5b-Joint Predictor Controls

- Date: 2026-06-09
- Scope: deterministic joint `future_delta_ee + future_gripper` predictor.

## Main Result

| branch | action MSE | SE(3) MSE | gripper MSE | transition acc |
| --- | ---: | ---: | ---: | ---: |
| task/proprio | 0.084648 | 0.043964 | 0.328752 | 0.009672 |
| visual patchpool | 0.040688 | 0.020486 | 0.161903 | 0.560270 |
| shuffled visual | 0.063790 | 0.038224 | 0.217181 | 0.330051 |

## Promotion Check

The visual joint predictor passes the deterministic promotion check:

```text
visual joint:       0.040688
modular bridge:     0.050333
best EEF-only prior 0.042090
```

It also passes visual attribution controls:

```text
visual < shuffled < task/proprio
```

for action MSE, SE(3) MSE, gripper MSE, and transition accuracy.

## Interpretation

The new representation direction is justified:

```text
future_delta_ee + future_gripper/event
```

The effect is not huge, but it is the first learned branch where adding the
gripper/event channel improves over the best learned EEF-only interface without
using oracle future EEF.

## Next

Move to a stochastic version:

```text
GeoMoCo-cVAE over joint EEF+gripper/event futures
```

The cVAE should be compared against this deterministic joint baseline, not only
against the older EEF-only baselines.

