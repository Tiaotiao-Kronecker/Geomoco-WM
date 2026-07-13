# Gate 2.3a Action-Aware Prior Vs Gate 2.2b

- Date: 2026-06-08
- Scope: compare aligned visual MSE-only future-motion training with
  action-aware visual future-motion training.

## Mean Metrics

| branch | future MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 | 0.00% |
| Gate 2.2b visual MSE-only | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 | 47.67% |
| Gate 2.3a action-aware | 0.000770 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 | 66.12% |
| Oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 | 100.00% |

## Decision Readout

Gate 2.3a improves the main downstream metric:

```text
0.043174 < 0.049547 action MSE
```

It also improves action MAE and gripper MSE:

```text
action MAE: 0.120370 -> 0.113432
gripper MSE: 0.222467 -> 0.177930
```

The direct-to-oracle action-MSE gap closure improves from `47.67%` to
`66.12%`.

## Interpretation

This is the first learned prior that explicitly optimizes for the executable
value of predicted future motion. The result supports the earlier hypothesis:
lower future-motion MSE alone is not enough; the future-motion representation
must be useful to the action interface.

The action-aware branch should now become the default deterministic learned
prior baseline before cVAE. The remaining oracle gap should be attacked with:

- lambda sweep for action-aware loss;
- multi-query / step-wise visual attention;
- gripper/contact auxiliary prediction;
- then multimodal cVAE / diffusion / flow priors.

## Relation To Oracle v2

Oracle v2 is now a side-track calibration line, not the immediate next run. Use
it when the learned prior gets close to the current oracle bound or when
progress stalls and we need to check whether the current oracle itself is too
weak.

The current mainline remains:

```text
Gate 2.3 action-aware deterministic prior
  -> stronger temporal / multimodal visual prior
  -> GeoMoCo-cVAE with validated visual-action route
  -> Oracle v2 calibration as needed
```
