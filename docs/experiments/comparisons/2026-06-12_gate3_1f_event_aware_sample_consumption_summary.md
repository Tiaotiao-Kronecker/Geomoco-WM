# Gate 3.1f Event-Aware Sample Consumption Summary

## Question

Does exposing predicted event identity/rank/probability let the action head use
the predicted event-mixture sample set?

## Result

Yes.

| branch | action MSE | gripper MSE |
| --- | ---: | ---: |
| Gate 3.0 real sample-set K=16 | 0.036675 | 0.164061 |
| Gate 3.1e anonymous top-4 | 0.038024 | 0.167432 |
| Gate 3.1f event-aware top-4 | 0.034767 | 0.150052 |

## Mainline Meaning

The earlier failure was not because predicted event mixtures were useless. It
was because the action head saw anonymous samples and had no explicit handle on
which event mode each sample came from.

Gate 3.1f supports the sharper GeoMoCo-WM story:

```text
visual context predicts event-structured future-motion modes
event-conditioned cVAE proposes future_delta_gripper samples
event-aware action head consumes the structured proposal set
```

## Decision

Promote event-aware top-4 as the current best deployable interface.

Next, run metadata ablations before adding a stronger flow/diffusion action
head.
