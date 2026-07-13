# Gate 3.7a Soft Event-Time Latent Conditional Decoder Plan

## Purpose

Test whether the close/open transition bottleneck is better handled by a soft
event-time latent rather than hard boundary routing, transition-weighted loss,
or checkpoint selection.

## Motivation

Recent gates showed:

```text
Gate 3.4: temporal MSE 0.034262, transition MSE 0.131311, sustain MSE 0.022542
Gate 3.6c: transition MSE improves to 0.126421, but temporal/sustain regress
```

Gate 3.6b/3.6c corrected the attribution: the transition improvement came from
transition-MSE checkpoint selection, not from candidate replacement. This means
the next branch should not keep tuning `transition_reserve`.

The first-principles problem is a hybrid timing problem:

```text
continuous EEF motion + discrete gripper/contact mode + close/open event time
```

Gate 3.7a therefore predicts a soft close/open time distribution and conditions
the temporal action decoder on that distribution.

## Minimal Design

Add a small optional mode to `MotionPriorActionHead`:

```text
event_time_conditioning_mode = none | soft_boundary
```

When enabled:

```text
1. predict close/open logits [B, 2, H+1]
   H regular classes = event step
   final class       = no event

2. compute softmax probabilities

3. embed the full probability distribution into the temporal action decoder
   context, not just the argmax index

4. train with:
   temporal action loss
   + small close/open CE auxiliary loss
```

Important distinction from earlier boundary-index branches:

```text
Earlier: predict hard boundary index and route residuals through argmax.
Gate 3.7a: use the soft event-time distribution as decoder conditioning.
```

This should avoid turning the branch into another brittle one-step boundary
localizer.

## Initial Full-Aligned Short Run

Keep the Gate 3.4 sample interface fixed:

```text
event_candidate_policy: topk
event_top_m: 4
num_samples: 16
sample_feature_mode: event_rank_prob
temporal_action_decoder_mode: sequence_mlp
temporal_action_loss_weight: 1.0
event_time_conditioning_mode: soft_boundary
event_time_loss_weight: 0.1
selection_metric: temporal_action_mse
epochs: 20
seeds: 7, 17
```

Do not select by transition MSE for the first branch. Gate 3.6c already showed
that transition-only selection buys transition by damaging sustain/overall.

## Expansion Criteria

Expand controls only if the full-aligned branch satisfies:

```text
temporal_action_mse <= 0.0344
temporal_action_transition_mse < 0.131311
temporal_action_sustain_mse <= 0.0230
```

This allows a tiny overall tolerance relative to Gate 3.4 but rejects another
3.6c-style transition/sustain trade-off.

## Controls If Positive

Use the same attribution ledger:

```text
decoder gain = same-checkpoint base/frozen temporal - richer decoder
prior gain = context-only/no-prior - full aligned
metadata gain = shuffled/rank-prob-only - full aligned
diversity gain = mean_repeated - full aligned
```

Matched controls:

```text
full event/rank/prob
shuffled event metadata
rank/prob-only
mean_repeated
context-only/no-prior
same decoder capacity but no motion-prior samples
```

## Diagnostics

Report:

```text
event_time_ce
close/open event fraction
close/open top-1 accuracy on valid event windows
close/open within-1 accuracy on valid event windows
mean event-time entropy
temporal_action_mse
temporal_action_transition_mse
temporal_action_sustain_mse
temporal_action_gripper_mse
```

Interpret event-time metrics as diagnostics, not promotion metrics. Promotion
still depends on action MSE and attribution controls.

## Positive Interpretation

If full aligned improves transition while preserving overall/sustain, and
controls show the gain depends on aligned samples/metadata:

```text
soft event-time conditioning is a useful structured decoder interface.
```

Then either expand controls or use the event-time distribution as an auxiliary
condition for a later action-chunk/diffusion policy.

## Negative Interpretation

If event-time metrics improve but action MSE does not:

```text
event-time prediction alone is not enough; contact geometry or stronger
action-sequence modeling is needed.
```

If action MSE improves equally in context-only:

```text
the branch is decoder-capacity gain, not GeoMoCo-WM prior gain.
```

If it repeats the 3.6c trade-off:

```text
soft event-time conditioning still cannot solve transition without damaging
sustain; move toward stronger receding-horizon action-chunk policy with strict
controls.
```
