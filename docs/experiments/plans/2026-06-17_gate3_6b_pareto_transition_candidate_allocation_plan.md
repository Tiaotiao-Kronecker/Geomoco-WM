# Gate 3.6b Pareto Transition Candidate Allocation Plan

## Purpose

Gate 3.6a proved that upstream transition candidate allocation is a real lever:

```text
Gate 3.4 transition MSE: 0.131311
Gate 3.6a transition MSE: 0.126421
```

But it regressed overall and sustain:

```text
Gate 3.4 temporal MSE: 0.034262
Gate 3.6a temporal MSE: 0.036391
Gate 3.4 sustain MSE: 0.022542
Gate 3.6a sustain MSE: 0.025539
```

Gate 3.6b keeps the decoder fixed and makes the transition-reserve policy more
Pareto-aware by using a more conservative reserve threshold and selecting best
epochs by overall temporal MSE.

## Hypothesis

The 0.15 reserve threshold forces too many transition candidates into top-M,
including cases where transition probability is only weak evidence. A higher
threshold should reduce sustain/overall damage while preserving part of the
transition gain.

## Sweep

Keep all else fixed:

```text
event_candidate_policy=transition_reserve
event_top_m=4
num_samples=16
sample_feature_mode=event_rank_prob
temporal_action_decoder_mode=sequence_mlp
temporal_action_loss_weight=1.0
```

Short-budget full-aligned sweep:

```text
threshold=0.25, selection_metric=temporal_action_mse
threshold=0.35, selection_metric=temporal_action_mse
```

Compare against:

```text
Gate 3.4 top-k reference
Gate 3.6a threshold=0.15, selection_metric=temporal_action_transition_mse
```

## Expansion Rule

Only expand controls if a full-aligned sweep branch satisfies:

```text
temporal_action_mse <= 0.0346
temporal_action_transition_mse < 0.131311
temporal_action_sustain_mse <= 0.0235
```

The `0.0346` and `0.0235` tolerances allow a small overall/sustain cost if the
transition gain remains substantial, but reject another 3.6a-style trade-off.

If no branch satisfies the expansion rule, archive as a threshold-sweep result
and move to candidate action-regret diagnostics.

## Controls If Expanded

Use the same attribution ledger:

```text
prior gain = context-only/no-prior - full aligned
metadata gain = shuffled/rank-prob-only - full aligned
diversity gain = mean_repeated - full aligned
decoder gain = same-checkpoint base/frozen temporal - richer decoder
```

Matched controls:

```text
full event/rank/prob
shuffled event metadata
rank/prob-only
mean_repeated
context-only/no-prior
```

## Interpretation

Positive:

```text
Transition candidate allocation can be tuned into a better Pareto point. Then
run full controls and consider calibrated reserve policies.
```

Negative:

```text
Fixed thresholding is too blunt. Next inspect whether reserved transition
candidates reduce action regret, not only whether their event label is useful.
```
