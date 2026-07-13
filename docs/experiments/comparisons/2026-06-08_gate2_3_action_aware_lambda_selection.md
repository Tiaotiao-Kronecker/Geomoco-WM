# Gate 2.3 Action-Aware Lambda Selection

- Date: 2026-06-08
- Status: completed
- Scope: select the default action-aware loss weight after Gate 2.3b.

## Selection Question

Gate 2.3a proved that a frozen action-decoder auxiliary loss helps. Gate 2.3b
asks whether the default should remain `lambda_action=0.010`, or whether a
weaker/stronger weight is better.

The primary metric is downstream action MSE after predicted future motion is
fed through the frozen Gate 1.6 oracle future-motion action decoder.

The guardrail metrics are future-motion validation MSE and decomposed future
translation/orientation error.

## Summary Table

| branch | future MSE | future trans L2 | future orient L2 | action MSE | action MAE | action trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct context | n/a | n/a | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 | 0.00% |
| Oracle future motion | n/a | n/a | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 | 100.00% |
| Gate 2.2b, lambda 0.000 | 0.000772 | 0.014441 | 0.050114 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 | 47.67% |
| Gate 2.3b, lambda 0.003 | 0.000796 | 0.015497 | 0.050467 | 0.045790 | 0.118458 | 0.015329 | 2.074289 | 0.189612 | 58.55% |
| Gate 2.3a, lambda 0.010 | 0.000770 | 0.016763 | 0.049942 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 | 66.12% |
| Gate 2.3b, lambda 0.030 | 0.000782 | 0.018767 | 0.050640 | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |

## Readout

`lambda_action=0.030` is the best action interface setting in this sweep. It
improves action MSE over the MSE-only visual branch by `15.05%` and over
`lambda_action=0.010` by `2.51%`.

The stronger weight also improves action MAE, action translation L2, rotation
geodesic error, and gripper MSE relative to `lambda_action=0.010`.

The tradeoff is motion-space translation:

```text
lambda 0.010 future trans L2: 0.016763
lambda 0.030 future trans L2: 0.018767
relative change: +11.96%
```

So `0.030` should not be described as a more geometrically accurate
future-motion predictor. It is a more action-executable future-motion
predictor under the current frozen decoder interface.

## Decision

Default next branch:

```text
lambda_action = 0.030
```

Use this for the action-value-prioritized deterministic baseline and for the
first cVAE/multimodal future-motion prior unless a later run shows motion
geometry collapse.

Reference branch:

```text
lambda_action = 0.010
```

Keep this as the balanced geometry reference, especially when comparing
stochastic or multi-query variants.

## Mainline Position

The mainline now becomes:

```text
Gate 2.2c visual controls
  -> Gate 2.3a action-aware prior
  -> Gate 2.3b lambda selection, default lambda 0.030
  -> Gate 2.4 multimodal / stochastic future-motion prior
  -> GeoMoCo-cVAE with validated visual-action route
  -> Oracle v2 upper-bound calibration if needed
```

Oracle v2 remains a side-track for upper-bound calibration, not the immediate
replacement for learned-prior work.

## Next Experiment Contract

For Gate 2.4 and cVAE:

1. Always compare against both `lambda_action=0.030` and `0.010`.
2. Report future-motion metrics and action metrics in the same table.
3. Treat action MSE improvement without future-motion guardrails as incomplete.
4. Add a gripper/contact diagnostic or auxiliary target because gripper gains
   are still not fully explained by EEF SE(3) motion alone.
