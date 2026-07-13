# Gate 3.1e Action Head Summary

## Question

Can the Gate 3 action head consume predicted event-mixture samples well enough
to beat the unconditional cVAE sample-set baseline?

## Result

Not yet.

| branch | action MSE | gripper MSE |
| --- | ---: | ---: |
| real unconditional sample-set K=16 | 0.036675 | 0.164061 |
| shuffled sample-set K=16 | 0.042633 | 0.186272 |
| predicted event top-2 | 0.038052 | 0.169784 |
| predicted event top-4 | 0.038024 | 0.167432 |

## Interpretation

Predicted event-mixture samples contain useful signal because both top-2 and
top-4 beat shuffled sample sets. However, the current action head does not turn
the strong Gate 3.1d best-of-K coverage into a stronger deployable action
prediction result.

The likely bottleneck is the sample-set interface:

```text
good futures exist in the set
but the action head sees only anonymous future-motion samples
and does not know event rank/probability/mode identity
```

## Decision

Do not keep sweeping top-M as the main path. Move next to an event-aware
consumption interface:

```text
future sample + event mode embedding + event probability/rank
```

or a set-wise planner/readout trained to compare candidates with event identity
available.
