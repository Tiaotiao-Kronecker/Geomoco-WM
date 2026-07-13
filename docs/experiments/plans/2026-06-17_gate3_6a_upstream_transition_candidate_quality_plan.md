# Gate 3.6a Upstream Transition Candidate Quality Plan

## Purpose

Gate 3.5b and 3.5c show that residual action adapters can improve overall MSE,
but do not fix close/open transition timing:

```text
Gate 3.4 temporal transition MSE:  0.131311
Gate 3.5b full transition MSE:     0.133339
Gate 3.5c predicted-gate transition MSE: 0.133308
```

So the next clean branch should move upstream: improve the predicted-event
candidate set before it reaches the action decoder, while keeping decoder
capacity fixed.

## Hypothesis

The current top-M event mixture may under-allocate candidate slots to transition
events when transition probability is real but not top-ranked. If transition
candidate quality is the bottleneck, a deployable transition-reserve candidate
policy should improve transition MSE without relying on a larger decoder.

## Minimal Candidate Policy

Keep the Gate 3.4 temporal action-sequence decoder and Gate 3.1f/g sample
metadata interface. Change only how top-M event candidates are selected.

Baseline:

```text
top_indices = topk(event_probs, M)
```

Gate 3.6a:

```text
top_indices = topk(event_probs, M)
if no selected candidate is a transition
   and max transition prob >= reserve_threshold:
       replace the lowest-prob selected candidate with the best transition class
sort selected candidates by probability again
```

Suggested first config:

```text
event_candidate_policy=transition_reserve
transition_reserve_threshold=0.15
event_top_m=4
num_samples=16
sample_feature_mode=event_rank_prob
temporal_action_decoder_mode=sequence_mlp
selection_metric=temporal_action_transition_mse
```

This is deployable because it uses only event-probe probabilities.

## Controls

If full aligned passes the promotion check, keep the same attribution ledger:

```text
decoder gain = same-checkpoint base/frozen temporal minus richer decoder
prior gain = context-only/no-prior minus full aligned
metadata gain = shuffled/rank-prob-only minus full aligned
diversity gain = mean_repeated minus full aligned
```

Matched controls:

```text
full event/rank/prob samples with transition_reserve
shuffled event metadata with transition_reserve
rank/prob-only with transition_reserve
mean_repeated with transition_reserve
context-only/no-prior with transition_reserve
```

Also compare to the original Gate 3.4 top-k candidate policy.

## Promotion Check

Before full controls, two-seed full aligned should satisfy:

```text
temporal_action_mse <= Gate 3.4 temporal MSE 0.034262
temporal_action_transition_mse <= Gate 3.4 temporal transition MSE 0.131311
```

If it fails transition, archive and stop. If it passes, run the matched
controls and usage audit.

## Interpretation

Positive:

```text
Transition candidate allocation was limiting the action decoder. Then tune or
calibrate event-probe candidate quality under controls.
```

Negative:

```text
The issue is not just missing transition labels in top-M. Next inspect event
candidate timing quality and candidate action regret, or consider a stronger
event-conditioned prior rather than more decoder-side repair.
```
