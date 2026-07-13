# Gate 2.5a Future-Gripper Visual Controls

- Date: 2026-06-09
- Scope: visual attribution and action-bridge value for predicted future
  gripper/event channels.

## Summary

| variant | gripper MSE | event acc | transition acc | bridge action MSE | bridge gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 0.324415 | 0.595893 | 0.014386 | 0.050046 | 0.318862 |
| visual patchpool | 0.172088 | 0.888486 | 0.634542 | 0.028987 | 0.173254 |
| shuffled visual patchpool | 0.233726 | 0.818344 | 0.481249 | 0.037060 | 0.229232 |

## Core Finding

Real DINO patch-pooled visual grounding is the best branch on all three
levels:

```text
future gripper regression
transition event fidelity
downstream action bridge value
```

The shuffled visual control is not collapsed to task/proprio, which suggests it
contains dataset/task/scene priors. However, the real visual branch is still
substantially better than shuffled visual:

```text
action MSE:   0.037060 -> 0.028987
gripper MSE: 0.229232 -> 0.173254
transition:  0.481249 -> 0.634542
```

## Relation To Oracle Bounds

| input | action MSE | gripper MSE |
| --- | ---: | ---: |
| GT future EEF only | 0.031474 | 0.184683 |
| GT future EEF + predicted visual gripper | 0.028987 | 0.173254 |
| GT future EEF + GT future gripper | 0.004202 | 0.000241 |

The predicted visual gripper channel is enough to beat the EEF-only oracle
interface, but it closes only a small part of the large GT-gripper gap.

## Mainline Consequence

Gate 2.5a validates the new representation direction:

```text
future_delta_ee + future_gripper/event
```

It does not yet validate a fully deployable learned prior. The next check must
remove the remaining oracle EEF privilege:

```text
predicted future EEF + predicted future gripper
```

If that passes, the cVAE output space should be upgraded to joint
EEF+gripper/event futures.

