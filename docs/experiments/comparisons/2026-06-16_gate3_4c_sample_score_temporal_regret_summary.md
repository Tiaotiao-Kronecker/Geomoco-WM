# Gate 3.4c Sample-Score Temporal Regret Summary

## Question

Can set-wise candidate-comparison supervision make GeoMoCo-WM K-sample diversity
more indispensable than it was in Gate 3.4?

Gate 3.4 already had useful attribution:

```text
full aligned beat shuffled metadata, rank/prob-only, and context-only/no-prior.
```

But its trained mean-repeated control was close:

```text
full aligned temporal MSE: 0.034262
mean_repeated temporal MSE: 0.034414
diversity gain: +0.000152
```

Gate 3.4c tried to enlarge that diversity gain by giving the action head an
explicit sample-score objective.

## Result

Negative/neutral under the short budget.

Mean over seeds 7 and 17:

| branch | temporal MSE | gripper MSE | transition MSE |
| --- | ---: | ---: | ---: |
| Gate 3.4 full aligned | 0.034262 | 0.149383 | 0.131311 |
| Gate 3.4 mean_repeated | 0.034414 | 0.149085 | 0.132199 |
| Gate 3.4 context-only/no-prior | 0.036642 | 0.156306 | 0.136922 |
| Gate 3.4c motion-regret scorer | 0.034779 | 0.154203 | 0.141620 |

The scorer branch regresses from Gate 3.4 on the primary temporal-action metric
and especially on transition windows.

## What Was Learned

The Gate 3.4b diagnosis remains correct:

```text
The full-aligned Gate 3.4 checkpoint uses K-sample distribution at runtime,
but training still does not force a large advantage over a model adapted to
mean-only inputs.
```

Gate 3.4c adds one more fact:

```text
Supervising sample weights with future-motion regret is not sufficient to
convert that runtime diversity usage into a stronger action-policy gain.
```

The likely failure mode is target mismatch. `motion_regret` compares candidate
future motions to oracle future motion, while the deployed metric is action
sequence MSE, with gripper transitions dominating the hard windows. A candidate
that is geometrically close on average may still be the wrong candidate for
close/open timing.

## Decision

Do not run the full Gate 3.4c control matrix unless revisiting a narrower
scorer ablation. The first full-aligned promotion check failed, so full controls
would mostly measure why a non-promoted branch failed.

Keep the implemented sample scorer as a diagnostic hook. It can support a later
`temporal_action_regret` target, but that should not block the mainline.

## Next Mainline

Move next to a small flow/diffusion residual decoder only under the same
attribution discipline:

```text
decoder gain = same-checkpoint base MSE - richer-decoder MSE
prior gain = context-only/no-prior MSE - full aligned MSE
metadata gain = metadata-control MSE - full aligned MSE
diversity gain = mean_repeated MSE - full aligned MSE
```

The pass condition should be stricter than Gate 3.4:

```text
1. improve full-aligned temporal/action MSE over Gate 3.4;
2. retain a clear prior gain over context-only/no-prior;
3. retain metadata gain over shuffled and rank/prob-only controls;
4. increase or at least preserve diversity gain over trained mean_repeated;
5. pass same-checkpoint usage audits: mean collapse hurts, permutation is
   invariant, batch mismatch fails.
```

If a richer residual decoder improves full aligned but context-only improves
equally, the gain belongs to decoder capacity rather than GeoMoCo-WM. If full
aligned remains uniquely better under these controls, attribution still stands.
