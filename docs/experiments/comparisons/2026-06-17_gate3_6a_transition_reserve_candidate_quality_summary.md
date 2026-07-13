# Gate 3.6a Transition-Reserve Candidate Quality Summary

## Question

After transition-constrained residual adapters failed, does improving upstream
event/transition candidate allocation help the transition bottleneck?

## Result

Yes for transition, no for overall default promotion.

Mean over seeds 7 and 17:

| branch | temporal MSE | transition MSE | sustain MSE |
| --- | ---: | ---: | ---: |
| Gate 3.4 top-k reference | 0.034262 | 0.131311 | 0.022542 |
| Gate 3.6a full transition-reserve | 0.036391 | 0.126421 | 0.025539 |
| shuffled event metadata | 0.037033 | 0.127638 | 0.026105 |
| rank/prob-only metadata | 0.037415 | 0.128118 | 0.026476 |
| mean repeated | 0.034992 | 0.127759 | 0.023820 |
| context-only/no-prior | 0.039660 | 0.131626 | 0.028587 |

## Attribution

Transition ledger:

```text
prior gain    = context-only - full aligned   = +0.005205
metadata gain = shuffled - full aligned       = +0.001217
metadata gain = rank/prob-only - full aligned = +0.001697
diversity gain = mean_repeated - full aligned = +0.001338
vs Gate 3.4 top-k reference                   = +0.004890
```

Overall ledger:

```text
prior gain    = context-only - full aligned   = +0.003268
metadata gain = shuffled - full aligned       = +0.000641
metadata gain = rank/prob-only - full aligned = +0.001023
diversity gain = mean_repeated - full aligned = -0.001399
vs Gate 3.4 top-k reference                   = -0.002129
```

The transition result is attribution-positive: aligned transition-reserve
candidates beat no-prior, shuffled metadata, rank/prob-only, and mean-repeated
controls on transition MSE. The overall result is not positive because
mean-repeated and the old Gate 3.4 top-k reference both beat full 3.6a on
overall temporal MSE.

## Post-hoc Attribution Correction

Gate 3.6b/3.6c later showed that `transition_reserve` did not trigger at
`top_m=4`, because top-4 already contained a transition candidate for every
train and validation window. A `topk` run with only
`selection_metric=temporal_action_transition_mse` exactly reproduced the 3.6a
metrics.

So the transition gain remains real, but its cause should be attributed to
transition-MSE checkpoint selection / early stopping rather than candidate
reserve replacement.

## Decision

Archive Gate 3.6a as:

```text
transition mechanism-positive
transition-selection positive
candidate-reserve attribution not supported post-hoc
overall/sustain trade-off negative
not promoted as default
```

Next step should not spend more budget on the current fixed-threshold reserve
rule at top-4. Candidate work should resume only with a policy that actually
changes candidate composition or scores candidate futures by downstream
transition/action regret.
