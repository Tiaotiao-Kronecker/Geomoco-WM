# Gate 3.11a On-Generator GeoMoCo Critic Audit Plan

## Purpose

Test the GraspGen-style idea in the GeoMoCo-WM setting without immediately
training a stronger diffusion or flow action generator.

This gate is an attribution audit, not a final-performance run:

```text
Can a lightweight critic trained on GeoMoCo generator samples identify
future-motion candidates that have lower downstream action regret?
```

## Motivation

GraspGen separates two roles:

```text
generator      -> produce many 6-DOF grasp candidates
discriminator  -> score candidates drawn from the generator's own distribution
```

For GeoMoCo-WM the analogous split is:

```text
GeoMoCo cVAE/event mixture -> K future_delta_ee + future_gripper/event candidates
on-generator critic        -> score candidates by downstream temporal-action regret
frozen action decoder      -> evaluate whether selected candidates improve action value
```

This avoids the Gate 3.9 risk where a strong direct action generator can learn
the task from visual/proprio context and erase the measurable contribution of
the GeoMoCo interface.

## Questions

Gate 3.11a answers three narrow questions:

1. Does the current GeoMoCo sample set contain candidates better than the
   current set-level temporal readout?
2. Can a small critic learn to rank generator-sampled candidates by frozen
   temporal-action regret?
3. Does the gain depend on aligned GeoMoCo/event metadata rather than generic
   training capacity or broken controls?

## Minimal Implementation

New script:

```text
scripts/train_on_generator_geomoco_critic.py
```

Inputs:

```text
Gate 3.4 temporal action-head checkpoint
event-conditioned GeoMoCo-cVAE checkpoint referenced by that action head
Gate 3.1b event probe checkpoint referenced by that action head
event-mode audit JSON referenced by those checkpoints
```

Training loop:

```text
1. Freeze cVAE, event probe, and Gate 3.4 temporal action head.
2. For each batch, sample predicted top-M event-mixture GeoMoCo candidates.
3. For each candidate, run the frozen temporal action decoder with that single
   candidate.
4. Label each candidate by temporal-action regret:

   regret_k = MSE(action_decoder(candidate_k), target_action_chunk)

5. Train a small MLP critic:

   score_k = critic(context, candidate_k, task/suite conditioning, sample metadata)

6. Supervise scores with soft CE or hard CE against the lowest-regret candidate.
```

Validation reports:

```text
set_temporal_mse
candidate_mean_mse
candidate_oracle_mse
oracle_gain_vs_set
candidate_best_vs_mean_gap
critic_selected_mse
critic_selected_gain_vs_set
critic_gain_vs_mean
critic_gap_to_oracle
critic_top1_accuracy
critic_selected_beats_set
transition/sustain group metrics
```

## Controls

Run the same frozen decoder and critic capacity under these candidate controls:

```text
real
  aligned generated futures plus aligned event/rank/prob metadata

mean_repeated
  repeat the per-window mean future motion across K slots

rank_prob_only
  zero event identity while keeping rank/prob metadata

shuffled_event_identity
  roll event identity across the batch while keeping rank/prob metadata

batch_mismatch
  roll whole future candidates and metadata across the batch

zero_sample_features
  remove event/rank/prob metadata from critic and single-candidate labeling
```

Primary attribution check:

```text
real critic_selected_mse < mean_repeated critic_selected_mse
real critic_selected_mse < shuffled_event_identity critic_selected_mse
real critic_selected_mse < rank_prob_only critic_selected_mse
real critic_selected_gain_vs_set > 0
real transition critic_selected_gain_vs_set > 0
```

If `candidate_oracle_mse` is much better than `set_temporal_mse` but
`critic_selected_mse` cannot improve over the set readout, the sample space is
useful but the critic is still too weak or missing labels.

If `candidate_oracle_mse` is not better than `set_temporal_mse`, the current
GeoMoCo generator does not contain the candidate value needed for this route.

## Short-Budget Execution

Seed 7 real branch first:

```bash
.venv/bin/python scripts/train_on_generator_geomoco_critic.py \
  --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt \
  --output-dir outputs/on_generator_geomoco_critic/gate3_11a_real_seed7 \
  --candidate-control real \
  --epochs 10 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --quiet
```

If the real branch shows positive `oracle_gain_vs_set` and non-trivial critic
selection, run controls:

```bash
for control in mean_repeated rank_prob_only shuffled_event_identity batch_mismatch zero_sample_features; do
  .venv/bin/python scripts/train_on_generator_geomoco_critic.py \
    --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt \
    --output-dir outputs/on_generator_geomoco_critic/gate3_11a_${control}_seed7 \
    --candidate-control ${control} \
    --epochs 10 \
    --batch-size 64 \
    --seed 7 \
    --device cuda \
    --quiet
done
```

If seed 7 passes the attribution check, repeat the real branch and the strongest
two controls on seed 17:

```bash
.venv/bin/python scripts/train_on_generator_geomoco_critic.py \
  --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed17/model.pt \
  --output-dir outputs/on_generator_geomoco_critic/gate3_11a_real_seed17 \
  --candidate-control real \
  --epochs 10 \
  --batch-size 64 \
  --seed 17 \
  --device cuda \
  --quiet
```

## Stop Rules

Stop and redesign the critic target if:

```text
candidate_oracle_mse clearly beats set_temporal_mse
but critic_selected_mse does not beat set_temporal_mse
```

Likely redesigns:

```text
transition-specific regret target
gripper/contact regret target
set-wise pairwise critic
critic labels augmented by contact/object-state/task-phase proxies
```

Stop generator/readout work and revisit candidate quality if:

```text
candidate_oracle_mse does not beat set_temporal_mse
```

That means the generated candidate set itself lacks useful downstream action
alternatives under the frozen Gate 3.4 decoder.

## Positive Interpretation

If the real branch beats controls and improves transition-sliced selection:

```text
The GeoMoCo event-mixture generator contains action-useful futures, and a
GraspGen-style on-generator critic can extract part of that value.
```

Then the next fair step is a stronger action generator with matched controls:

```text
direct_visual generator
vs
GeoMoCo-conditioned generator
```

Do not skip that replacement audit; otherwise any gain may come from the strong
action generator rather than GeoMoCo.
