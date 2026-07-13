# Gate 3.5c Transition-Constrained Post-Hoc Residual Summary

## Question

Can the Gate 3.5b post-hoc residual adapter be made transition-useful by
gating the residual with deployable predicted transition probability?

## Result

No. It keeps a small overall gain, but fails the transition promotion check.

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | overall MSE | transition MSE | interpretation |
| --- | ---: | ---: | --- |
| Gate 3.4 temporal reference | 0.034262 | 0.131311 | previous promoted reference |
| Gate 3.5b post-hoc adapter | 0.033374 | 0.133339 | better overall, worse transition |
| Gate 3.5c predicted transition gate | 0.033810 | 0.133308 | still better overall than 3.4, still transition-negative |
| Gate 3.5c frozen temporal inside same checkpoints | 0.034135 | 0.130916 | adapter worsens transition relative to its frozen base |

Gate statistics:

```text
residual_gate_mode = predicted_transition_prob
residual_leak = 0.05
mean gate = 0.274131
min gate = 0.050028
max gate = 0.993391
```

## Gain Ledger

For the full-aligned 3.5c branch:

```text
decoder gain overall    = frozen temporal - adapter = +0.000325
decoder gain transition = frozen temporal - adapter = -0.002392
```

The sign split matters: the adapter is still learning useful generic repair,
but the transition-specific target is harmed.

## Stop Decision

The planned full-aligned promotion criteria were:

```text
adapter_transition_mse <= Gate 3.4 temporal transition MSE 0.131311
adapter_mse <= Gate 3.4 temporal MSE 0.034262
decoder_gain_transition_mse >= 0
```

3.5c only passes the overall MSE condition. It fails both transition conditions:

```text
adapter_transition_mse = 0.133308 > 0.131311
decoder_gain_transition_mse = -0.002392
```

Therefore the full attribution control matrix was not run. The controls remain
the required ledger for the next positive decoder:

```text
decoder gain = same-checkpoint base/frozen temporal - richer decoder
prior gain = context-only/no-prior - full aligned
metadata gain = shuffled/rank-prob-only - full aligned
diversity gain = mean_repeated - full aligned
```

## Interpretation

Gate 3.5c narrows the diagnosis:

```text
1. Generic post-hoc residual capacity is useful for overall MSE.
2. Deployable transition-probability gating does not make that capacity solve
   close/open transition timing.
3. The bottleneck is now more likely upstream event/transition candidate quality
   than another small residual adapter.
```

The next mainline should improve the event/transition proposal quality itself:
better transition confidence calibration, transition-rank allocation, or
transition-specific candidate generation. Any subsequent positive result must
still carry the decoder/prior/metadata/diversity controls.

## Decision

Archive Gate 3.5c as:

```text
overall weak-positive
transition-negative
adapter-family warning
upstream-candidate-quality pivot point
```
