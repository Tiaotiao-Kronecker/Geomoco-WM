# Gate 2.1 Conditioned Future-Motion Prior Vs Bounds

- Date: 2026-06-07
- Scope: compare the suite/task-conditioned deterministic future-motion prior
  against zero motion, Gate 2 context-only prior, direct context, and oracle
  future motion.

## Branches

| branch | prior input | action-decoder input | role |
| --- | --- | --- | --- |
| zero future motion | none | context + zeros | empty-channel sanity baseline |
| Gate 2 context-only prior | context/proprio | context + predicted future motion | first learned prior |
| Gate 2.1 suite/task prior | context/proprio + suite/task one-hot | context + predicted future motion | tested conditioned prior |
| direct context | n/a | context only | lower-bound policy interface without future motion |
| oracle future motion | n/a | context + GT future EEF delta | upper/interface bound |

## Mean Metrics

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero future motion | 0.001896 | 0.110711 | 0.199303 | 0.027867 | 2.619721 | 0.373234 |
| Gate 2 context-only prior | 0.001027 | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Decision Readout

Task/suite metadata helps:

- `9.59%` lower future-motion MSE than Gate 2.
- `10.81%` lower downstream action MSE than Gate 2.

But it is not enough:

- Gate 2.1 action MSE is still `9.83%` worse than direct context.
- The learned prior still fails the promotion criterion:

```text
learned future motion action MSE < direct context action MSE
```

## Interpretation

The failure is now narrower than Gate 2. The model is no longer merely missing
task identity; even with suite/task metadata, a deterministic future EEF-delta
prediction is still not executable enough for the frozen action decoder.

This supports the original GeoMoCo-WM direction: future motion needs stronger
grounding and/or stronger action-aware supervision. A better future-motion MSE
alone is still insufficient.

## Next Mainline

Proceed to a value-carrying prior rather than a larger plain MLP:

```text
Gate 2.2: add visual grounding token / DINO feature conditioning
Gate 2.3: add action-aware auxiliary loss or joint prior-decoder training
Gate 2.4: model gripper/contact as a separate auxiliary branch
```

