# Gate 2.4h-c Event Alignment Summary

- Date: 2026-06-09
- Scope: cVAE prior mean, random samples, event oracle, action oracle, structured
  oracle, and Gate 2.4d ScoreNet selection on transition labels.

## Summary Table

Mean over seeds 7 and 17:

| method | event acc | transition acc | step within 1 | deployable |
| --- | ---: | ---: | ---: | --- |
| prior mean | 0.836666 | 0.421782 | 0.179168 | yes |
| random sample mean | 0.835440 | 0.416745 | 0.172950 | yes |
| event oracle best-of-K | 0.849049 | 0.452092 | 0.219359 | no |
| flat action oracle best-of-K | 0.841919 | 0.412110 | 0.161534 | no |
| SE(3)+gripper oracle best-of-K | 0.838495 | 0.405231 | 0.161324 | no |
| ScoreNet argmax | 0.839062 | 0.408688 | 0.175082 | yes |

## What This Means

The current visual cVAE samples carry some event signal, but the signal is not
strong enough yet:

- event oracle best-of-K is above prior mean, so useful event-aligned samples
  sometimes exist;
- transition timing coverage is low, so many close/open phase changes are not
  represented precisely in the sampled futures;
- flat ScoreNet is not an event readout, and its selected samples do not close
  the event-oracle gap.

This explains why previous readout improvements were small: the model is not
only choosing among equally good event candidates. It is often missing precise
transition timing, and when good event candidates exist, the current scorer is
not trained to care about them.

## Mainline Decision

Gate 2.4h-d should be a small controlled diagnostic, not a big method jump:

```text
try weak event-aware ScoreNet auxiliary supervision
keep flat action MSE as the promotion metric
require event metrics to improve without action-MSE regression
```

If this fails, the bottleneck is upstream: improve generated future-motion /
decoded-action event fidelity before adding more scorer machinery.
