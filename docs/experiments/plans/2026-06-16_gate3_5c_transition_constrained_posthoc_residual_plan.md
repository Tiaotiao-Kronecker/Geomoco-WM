# Gate 3.5c Transition-Constrained Post-Hoc Residual Plan

## Purpose

Gate 3.5b showed that a frozen post-hoc residual adapter improves overall MSE,
but mostly as generic action repair:

```text
Gate 3.5b full adapter MSE:        0.033374
Gate 3.4 temporal MSE:             0.034262
Gate 3.5b full transition MSE:     0.133339
Gate 3.4 temporal transition MSE:  0.131311
context-only adapter MSE:          0.033728
```

So the next step should not simply increase adapter capacity. Gate 3.5c keeps
the same frozen Gate 3.4 post-hoc setup, but constrains residual application to
transition-likely windows.

## Hypothesis

If residual capacity is useful but too generic, a deployable transition gate
should:

```text
preserve most overall decoder gain;
improve or at least not regress transition MSE;
reduce context-only catch-up;
keep full aligned better than metadata/diversity controls.
```

If transition-constrained residual still fails, the bottleneck is likely
upstream event/transition candidate quality rather than residual decoder
capacity.

## Minimal Design

Use the existing frozen Gate 3.4 checkpoint and adapter:

```text
raw_residual = Adapter(frozen_features, frozen_temporal_actions)
adapter_actions = frozen_temporal_actions + gate * raw_residual
```

Gate options:

```text
predicted_transition_prob:
  deployable. Use the event probe top-M probabilities mapped to cVAE event
  classes. gate = sum prob over transition event labels.

oracle_transition:
  diagnostic upper-bound only. Use true event labels.

none:
  Gate 3.5b behavior.
```

Use a small leak so non-transition windows can still receive a tiny correction:

```text
effective_gate = leak + (1 - leak) * transition_prob
```

First run:

```text
residual_gate_mode=predicted_transition_prob
residual_gate_threshold=None
residual_leak=0.05
selection_metric=adapter_transition_mse
```

## Attribution Controls

Keep the same controls as Gate 3.5b:

```text
full event/rank/prob samples
shuffled event metadata
rank/prob-only
mean_repeated
context-only/no-prior
eval-time mean collapse
permutation sanity
batch mismatch
```

Compute:

```text
decoder gain = frozen temporal_actions MSE - adapter_actions MSE
prior gain = context-only/no-prior adapter MSE - full aligned adapter MSE
metadata gain = metadata-control adapter MSE - full aligned adapter MSE
diversity gain = mean_repeated adapter MSE - full aligned adapter MSE
```

## Promotion Check

Before full controls, full aligned should satisfy:

```text
adapter_transition_mse <= Gate 3.4 temporal transition MSE 0.131311
adapter_mse <= Gate 3.4 temporal MSE 0.034262
decoder_gain_transition_mse >= 0
```

If full aligned fails transition, stop or run only oracle-transition diagnostic.

## Interpretation

Positive:

```text
Residual action modeling can help when gated by deployable transition
confidence. Then expand controls and consider improving the event-confidence
source.
```

Negative:

```text
The adapter is not the main transition lever. Move upstream to event/transition
candidate quality: better event probe calibration, transition-rank allocation,
or transition-specific candidate generation.
```

