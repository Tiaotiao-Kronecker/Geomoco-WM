# Event-Time Latent vs Motion-Prior-Conditioned Action-Chunk Policy

Date: 2026-06-20

## Context

The current GeoMoCo-WM mainline has moved past the question of whether the
visual motion prior is useful. The stronger current bottleneck is
close/open transition timing inside the action chunk:

```text
Gate 3.4 temporal MSE:            0.034262
Gate 3.4 transition MSE:          0.131311
Gate 3.4 sustain MSE:             0.022542
Gate 3.6c transition-selected MSE: 0.126421 transition, but 0.036391 overall
```

Gate 3.6b/3.6c corrected the most recent attribution: the transition gain was
caused by selecting checkpoints with `temporal_action_transition_mse`, not by
`transition_reserve` candidate replacement. The fixed reserve rule is inert at
top-4 because top-4 already contains a transition candidate for every train and
validation window.

The next design question is whether to:

```text
A. model event time explicitly with a small latent/conditional decoder; or
B. jump to a stronger motion-prior-conditioned diffusion/action-chunk policy.
```

## What The Timing Problem Is

The failure is not simply knowing whether a close/open transition exists. The
event prior already proposes transition candidates. The problem is the exact
in-chunk timing:

```text
GT:     open open open close close close close close
early:  open open close close close close close close
late:   open open open open close close close close
```

Being early or late by one or two steps can raise offline transition MSE and
can also cause real execution failure. This is a hybrid-system problem:

```text
continuous EEF motion + discrete gripper/contact mode + precise event time
```

## Option A: Event-Time Latent + Conditional Decoder

Model a soft close/open event-time latent:

```text
p(close_step = 0..H or no_close | context, GeoMoCo samples, event metadata)
p(open_step  = 0..H or no_open  | context, GeoMoCo samples, event metadata)
```

Then condition the temporal action decoder on this distribution rather than
hard-routing through an argmax boundary.

### Advantages

- Directly targets the current bottleneck: close/open timing.
- Keeps attribution clean: prior, metadata, diversity, and decoder gains can
  still be measured with the existing ledger.
- Small-budget implementation is plausible because `close_step/open_step`
  labels already exist in the event-mode audit JSON.
- Provides interpretable diagnostics: CE, within-1 accuracy, entropy,
  calibration, transition/sustain Pareto.
- Preserves the GeoMoCo-WM story: the world-motion prior proposes event/motion
  futures, and the action decoder consumes structured timing evidence.

### Risks

- Can collapse back into a brittle boundary-localizer if implemented with hard
  argmax routing.
- If visual/motion samples do not contain enough contact evidence, event-time
  prediction may remain noisy.
- It mostly addresses gripper transition timing, not all continuous contact
  geometry.
- Demonstration style can make multiple close/open timings plausible for the
  same state, so hard one-hot supervision may be noisy.

### Guardrails

- Use the predicted distribution as a continuous decoder condition.
- Do not initially route actions through a single predicted boundary index.
- Report both event-time quality and action metrics.
- Preserve context-only, shuffled metadata, rank/prob-only, mean-repeated, and
  same-capacity controls if the full branch is positive.

## Option B: Motion-Prior-Conditioned Diffusion / Action-Chunk Policy

Train a stronger policy to directly model:

```text
p(action_chunk | observation, GeoMoCo samples, event/rank/prob metadata)
```

This can be diffusion-style, ACT-style action chunking, or a mode+residual
action policy such as a behavior transformer.

### Advantages

- More directly aligned with a final deployable policy.
- Naturally models multi-modal action sequences, gripper timing, and continuous
  motion together.
- Receding-horizon execution can reduce sensitivity to a single predicted
  boundary inside a long chunk.
- Avoids manually specifying the close/open latent form.

### Risks

- Much higher attribution risk: a strong decoder may solve the task from
  context/visual features alone and make GeoMoCo samples incidental.
- Larger compute and tuning budget.
- Harder to debug: if it improves, it may be unclear whether the gain came from
  motion prior, event metadata, sample diversity, or decoder capacity.
- Offline MSE alone may be especially misleading for a strong action policy
  unless paired with rollout/success metrics.

## Decision

Proceed with Option A first:

```text
Gate 3.7a = minimal soft event-time latent + conditional temporal decoder
```

Treat A as a diagnostic structural bridge, not as the final policy. If it
improves transition without damaging sustain/overall, it validates event-time
conditioning as the right interface. If it fails, it becomes useful evidence
for moving to B, possibly using the event-time distribution as an auxiliary
condition or diagnostic for the stronger policy.

