# Gate 2 Learned Future-Motion Prior Vs Bounds

- Date: 2026-06-06
- Scope: compare the first deterministic learned future-motion prior against
  zero future motion, direct context, and oracle future motion.

## Bounds

| branch | action-decoder input | role |
| --- | --- | --- |
| direct context | current context only, no future-motion input | lower-bound action decoder without future-motion input |
| zero future motion | current context plus an all-zero future-motion vector | sanity baseline for the frozen oracle action decoder |
| learned future motion | current context plus `FutureMotionPredictor(context)` | deterministic context-only future-motion prior being tested |
| oracle future motion | current context plus GT future EEF delta | GT future-motion upper/interface bound |

In this comparison, "the learned prior" or "it" means:

```text
FutureMotionPredictor(context/proprio) -> predicted future EEF delta
```

The learned prior is useful only if its downstream action metrics move between
the direct-context lower bound and the oracle-future upper/interface bound. A
good ordering would be:

```text
zero future motion > direct context > learned future motion > oracle future motion
```

The actual ordering is:

```text
zero future motion > learned future motion > direct context > oracle future motion
```

Therefore, the learned prior is better than feeding an empty future-motion
channel into the frozen decoder, but it is not yet an effective policy
intermediate variable.

## Mean Downstream Action Metrics

| branch | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct context | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| zero future motion | 0.110711 | 0.199303 | 0.027867 | 2.619721 | 0.373234 |
| learned future motion | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| oracle future motion | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Readout

The deterministic prior beats zero future motion, but it does not beat direct
context. This means the first learned prior is not yet good enough as a policy
interface, even though its future-motion prediction loss improves over a zero
baseline.

The important diagnosis:

```text
prediction MSE improves, downstream action value does not.
```

## Decision

Keep the Gate 1.6 geodesic replacement as the oracle interface bound. Treat
Gate 2 deterministic prior as a diagnostic baseline, not a promoted method.

Next experiment should improve the learned prior before cVAE claims:

```text
Gate 2.1: task/suite conditioning + stronger temporal predictor
Gate 2.2: visual grounding token / DINO feature conditioning
Gate 2.3: action-aware auxiliary loss or joint prior-decoder training
```

The first success criterion for a learned prior is not only future-motion MSE.
It must move downstream action metrics between the direct-context lower bound
and oracle-future upper/interface bound.
