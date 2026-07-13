# Gate 3.8a Controlled Small Action-Chunk Decoder Plan

## Purpose

Gate 3.7a showed that the minimal soft event-time latent is not a useful
conditioning signal in its current form. The high event-time accuracy was mostly
no-event accuracy, while positive close/open timing stayed weak and the action
decoder regressed.

Gate 3.8a therefore moves one small step toward a deployable action-chunk
decoder without changing the GeoMoCo-WM prior interface.

## Design

Add a controlled temporal action decoder mode:

```text
temporal_action_decoder_mode = temporal_transformer
```

Keep fixed:

```text
event_candidate_policy = topk
event_top_m = 4
num_samples = 16
sample_feature_mode = event_rank_prob
future_input_control = real
selection_metric = temporal_action_mse
temporal_action_loss_weight = 1.0
no event-time latent
no flow residual
no candidate policy changes
```

The new decoder should still emit:

```text
temporal_actions [B,H,A]
```

and should reuse the existing temporal-action loss, repeated evaluation, and
group metrics. The only intended mechanism change is that action-step tokens
interact through a tiny temporal transformer before per-step action prediction.

## Why This Is Different From Gate 3.4

Gate 3.4 `sequence_mlp` uses a global context token plus learned step queries,
then predicts each step mostly independently:

```text
context + step_query_h -> action_h
```

Gate 3.8a lets the predicted action-step representation see neighboring steps:

```text
context-conditioned step tokens -> small TransformerEncoder -> action chunk
```

This targets close/open timing as a trajectory property, not as a separately
predicted boundary index.

## Short-Budget Stop Rule

First run seed 7 full aligned only. Compare against the Gate 3.4 seed-7
full-aligned reference:

```text
Gate 3.4 seed7 temporal_action_mse            = 0.036502
Gate 3.4 seed7 temporal_action_transition_mse = 0.137827
Gate 3.4 seed7 temporal_action_sustain_mse    = 0.023501
```

Continue only if Gate 3.8a seed 7 satisfies:

```text
temporal_action_mse < 0.036502
temporal_action_transition_mse < 0.137827
temporal_action_sustain_mse <= about 0.0236
```

If not, stop without seed 17 or controls.

## Controls If Positive

If seed 7 passes, run seed 17. If the two-seed result remains positive, expand
the same attribution controls:

```text
full event/rank/prob
shuffled event metadata
rank/prob-only
mean_repeated
context-only/no-prior
same decoder capacity but no motion-prior samples
```

Keep the attribution ledger:

```text
decoder gain   = Gate 3.8a full aligned - Gate 3.4 temporal baseline
prior gain     = context-only/no-prior - full aligned
metadata gain  = shuffled/rank-prob-only - full aligned
diversity gain = mean_repeated - full aligned
```

Use positive MSE reductions as gains when reporting.

## Negative Interpretation

If full aligned seed 7 fails, the conclusion is narrow:

```text
a tiny temporal transformer action-chunk decoder is not enough under the fixed
Gate 3.4 event/rank/prob sample interface.
```

Do not interpret that as evidence against larger action-chunk policies or
diffusion/flow policies. It only closes the cheapest controlled step after the
Gate 3.7a soft event-time latent negative.
