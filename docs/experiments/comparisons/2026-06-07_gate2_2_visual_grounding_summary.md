# Gate 2.2 Visual Grounding Summary

- Date: 2026-06-07
- Scope: compare task/proprio learned priors, global DINO visual grounding, and
  patch-pooled DINO cross-attention grounding.

## Mean Metrics

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2 context-only prior | 0.001027 | 0.081291 | 0.166503 | 0.022733 | 2.189888 | 0.296775 |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| Gate 2.2a DINO global visual prior | 0.000801 | 0.053628 | 0.128207 | 0.016285 | 2.031589 | 0.229947 |
| Gate 2.2b patch cross-attention prior | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Decision Readout

Both visual priors beat direct context:

```text
Gate 2.2a action MSE: 0.053628 < 0.066010
Gate 2.2b action MSE: 0.049547 < 0.066010
```

Patch cross-attention improves over global visual tokens:

```text
0.049547 < 0.053628
```

The direct-to-oracle action-MSE gap closure improves from:

```text
Gate 2.2a: 35.85%
Gate 2.2b: 47.67%
```

## Interpretation

This is the clearest mechanism result so far. The old GeoMoCo concern was that
EEF-centric latent/state factors were too weak to help policy. Gate 2.2 shows
that when future EEF motion is grounded in visual information, the learned
motion prior becomes useful to the action interface.

The result supports this working hierarchy:

```text
oracle future motion
  > patch visual grounded learned future motion
  > global visual grounded learned future motion
  > direct context
  > task/proprio-only learned future motion
  > zero future motion
```

## Required Controls Before cVAE

Do not jump directly to cVAE claims until these are run:

```text
shuffled visual feature control
camera ablation: agentview-only / eye-in-hand-only / two-camera
optional stronger action-aware objective
```

After controls, use the best visual grounding route as the conditioning path
for visual-grounded GeoMoCo-cVAE.

