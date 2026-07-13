# Gate 2.2a Visual Prior Vs Bounds

- Date: 2026-06-07
- Scope: compare the DINOv2 global visual future-motion prior against zero
  future motion, Gate 2 context-only prior, Gate 2.1 suite/task prior, direct
  context, and oracle future motion.

## Mean Metrics

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero future motion | 0.001896 | 0.110711 | 0.199303 | 0.027867 | 2.619721 | 0.373234 |
| Gate 2 context-only prior | 0.001027 | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| Gate 2.2a DINO global visual prior | 0.000801 | 0.053628 | 0.128207 | 0.016285 | 2.031589 | 0.229947 |
| direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Promotion Criterion

The learned prior should beat direct context through the frozen action-decoder
interface:

```text
learned future motion action MSE < direct context action MSE
```

Gate 2.2a passes:

```text
0.053628 < 0.066010
```

It also improves the physical action metrics:

```text
translation_m_l2: 0.016285 < 0.019024
rotation_geodesic_deg: 2.031589 < 2.233651
gripper_mse: 0.229947 < 0.252545
```

## Decision Readout

This is a positive visual-grounding result.

Compared with Gate 2.1, visual features reduce:

- future-motion MSE by `13.68%`;
- downstream action MSE by `26.03%`.

Compared with direct context, visual learned future motion reduces downstream
action MSE by `18.76%` and closes `35.85%` of the direct-to-oracle action-MSE
gap.

## Interpretation

Task identity was not enough in Gate 2.1. Adding DINOv2 global visual features
pushes the learned future-motion prior into the action-value interval between
direct context and oracle future motion.

This supports the working thesis:

```text
GeoMoCo-WM needs visual grounding for future motion to become executable.
```

## Required Follow-Up Controls

Before claiming a final method result, run:

```text
shuffled visual feature control
agentview-only / eye-in-hand-only / two-camera ablation
patch-token cross-attention Gate 2.2b
```

Only after those controls should the project move to visual-grounded cVAE.

