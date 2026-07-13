# Gate 3.5a Small Residual Flow Decoder Summary

## Question

After Gate 3.4c failed as a motion-regret scorer, can a small residual
flow-style action decoder improve the Gate 3.4 temporal action decoder while
preserving attribution?

## Result

No, not in this minimal joint-training form.

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | action MSE | gripper MSE | transition MSE |
| --- | ---: | ---: | ---: |
| Gate 3.4 temporal | 0.034262 | 0.149383 | 0.131311 |
| Gate 3.5a temporal | 0.035392 | 0.155863 | 0.136396 |
| Gate 3.5a flow | 0.035621 | 0.156430 | 0.137830 |

The flow residual is worse than its own temporal base:

```text
decoder gain = temporal_actions - flow_actions
             = 0.035392 - 0.035621
             = -0.000229
```

So the branch fails before prior/metadata/diversity attribution controls.

## Interpretation

The failure mode is different from Gate 3.4c.

Gate 3.4c failed because motion-regret candidate scoring was not aligned enough
with downstream action/gripper-transition value.

Gate 3.5a fails because the residual flow branch does not produce a better
action trajectory than the temporal decoder. Since the temporal branch in the
same jointly trained checkpoint also regresses from Gate 3.4, the flow
objective likely perturbs shared representation learning or adds noisy residual
pressure.

## Decision

Do not run the full attribution matrix for Gate 3.5a. The first full-aligned
promotion check failed:

```text
flow_action_mse 0.035621 > Gate 3.4 temporal_action_mse 0.034262
```

The attribution controls remain the right standard, but they should only be
spent after the full-aligned richer decoder beats Gate 3.4.

## Next Mainline

The cleanest next small step is not a larger black-box diffusion policy. It is:

```text
Gate 3.5b: post-hoc residual adapter over frozen Gate 3.4
```

Purpose:

```text
Freeze the Gate 3.4 action head.
Train only a small residual adapter on top of frozen temporal_actions.
Keep the same predicted top-4 event/rank/prob inputs.
```

This isolates whether residual action modeling is useful without letting the
new objective degrade the temporal decoder. If frozen-adapter full aligned
beats Gate 3.4, then run the standard controls:

```text
decoder gain = frozen temporal_actions MSE - residual output MSE
prior gain = context-only/no-prior residual MSE - full aligned residual MSE
metadata gain = shuffled/rank-prob-only residual MSE - full aligned residual MSE
diversity gain = mean_repeated residual MSE - full aligned residual MSE
```

If frozen-adapter full aligned also fails, the bottleneck is probably not
decoder capacity; go back upstream to transition/event candidate quality.
