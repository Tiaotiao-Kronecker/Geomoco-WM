# Gate 3.4c Sample-Score Temporal/Regret Supervision Plan

## Purpose

Gate 3.4b showed that the full-aligned temporal decoder uses K-sample structure
at runtime, but the separately trained `mean_repeated` control remains close.
Gate 3.4c tests whether sample-distribution use can be made more causally
necessary under matched training controls.

## Minimal Design

Add an optional candidate scorer to `MotionPriorActionHead`:

```text
sample_score_mode=action_regret
```

When enabled:

```text
1. each sample token receives a scalar score;
2. softmax(score) is used as the set aggregation weight;
3. the scorer receives a candidate-comparison auxiliary loss.
```

This keeps the decoder small and controlled. It is not a flow/diffusion policy
decoder.

## First Target

Use the cheap, stable target first:

```text
sample_score_target=motion_regret
```

For each sample:

```text
regret_k = mean_square(sample_motion_k - oracle_future_motion)
target_probs = softmax(-regret / temperature)
loss = CE(target_probs, scorer_logits)
```

This should teach the action head to prefer samples that are closer to the
oracle future-motion target. It does not use action labels to select samples,
so attribution remains tied to the motion-prior interface.

Optional later target:

```text
sample_score_target=temporal_action_regret
```

This is more directly action-valued but more expensive and more self-referential
because it scores single-sample temporal-action predictions.

## Controls

Keep the Gate 3.4 attribution matrix:

```text
full aligned event/rank/prob
shuffled event metadata
rank/prob-only
trained mean_repeated
context-only/no-prior same-capacity decoder
eval-time mean collapse via usage audit
permutation and batch-mismatch sanity
```

For `context_only`, keep `sample_score_mode=none` because there are no candidate
samples to score. This remains the same-capacity no-prior decoder control.

## Primary Metrics

Promotion metrics:

```text
temporal_action_mse
temporal_action_transition_mse
temporal_action_gripper_mse
```

Attribution metrics:

```text
full aligned - trained mean_repeated gap
full aligned - shuffled metadata gap
full aligned - rank/prob-only gap
full aligned - context-only gap
usage-audit eval-time mean collapse gap
```

Scorer diagnostics:

```text
sample_score_loss
sample_score_top1_accuracy
sample_score_expected_regret
sample_score_expected_vs_best_gap
sample_score_entropy
```

## Short-Budget Run

Start with:

```text
sample_score_loss_weight=0.1
sample_score_loss_type=soft_ce
sample_score_temperature=0.05
epochs=20
seeds=7,17
```

Decision:

```text
If full aligned improves and the trained mean_repeated gap grows, Gate 3.4c is
a cleaner diversity-usage step.

If full aligned regresses or controls improve equally, sample-score supervision
is not enough; then test a small flow/diffusion residual decoder with the same
controls.
```
