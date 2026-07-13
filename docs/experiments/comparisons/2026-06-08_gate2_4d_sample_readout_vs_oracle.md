# Gate 2.4d Sample Readout Vs Oracle Best-Of-K

- Date: 2026-06-08
- Status: completed
- Scope: compare deployable cVAE sample readouts against prior mean, random
  samples, and non-deployable oracle best-of-K.

## Summary

| branch | deployable? | action MSE | gripper MSE | note |
| --- | --- | ---: | ---: | --- |
| Gate 2.4c prior mean | yes | 0.040931 | 0.167670 | stable cVAE baseline |
| Gate 2.4c random sample mean | yes | 0.041183 | 0.167995 | naive stochastic readout |
| Gate 2.4d ScoreNet argmax | yes | 0.040201 | 0.165270 | first learned readout |
| Gate 2.4d ScoreNet soft motion | yes | 0.040709 | 0.166785 | lower-variance but weaker |
| Oracle best-of-K action | no | 0.036895 | 0.152010 | GT-selected upper-bound diagnostic |

## Readout Gap

The lightweight scorer improves over prior mean:

```text
0.040931 -> 0.040201
relative improvement: 1.78%
```

It also beats random sample mean:

```text
0.041183 -> 0.040201
relative improvement: 2.38%
```

But it only closes part of the oracle readout gap:

```text
prior mean: 0.040931
oracle best-of-K: 0.036895
ScoreNet argmax: 0.040201
gap closed: 18.09%
```

## Decision

Gate 2.4d validates the direction:

- useful samples can be selected without GT at test time;
- action MSE improves in both seeds;
- gripper MSE improves slightly;
- the method remains attribution-clean because the action decoder is frozen and
  deterministic.

However, the readout is still weak relative to oracle best-of-K. The next
mainline should not immediately switch to a multimodal action head. It should
first strengthen the readout with:

1. gripper/contact/executability labels or diagnostics;
2. stronger ranking losses or hard-negative mining;
3. optional scorer calibration using future-motion and action-value hybrid
   targets.
