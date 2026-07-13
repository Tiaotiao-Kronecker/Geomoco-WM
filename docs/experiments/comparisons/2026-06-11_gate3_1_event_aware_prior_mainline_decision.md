# Gate 3.1 Event-Aware Prior Mainline Decision

## Why Gate 3.1

Gate 3.0c showed:

```text
aligned sample-set diversity helps
generic shuffled diversity hurts
transition windows are much harder than no-transition windows
mean replacement hurts transition windows more
```

Therefore the next bottleneck is not K or set aggregation. The next bottleneck
is whether the future-motion prior can expose aligned modes, especially gripper
transition/event timing modes.

## Mainline Direction

Move to:

```text
mode-structured / event-aware future-motion prior
```

This means:

```text
future_delta_gripper samples should vary along explicit event/timing modes,
not only along an unstructured Gaussian latent.
```

## Minimal First Version

Use weak action-derived event labels:

```text
transition_close
transition_open
mixed_transition
sustain_close
sustain_open
hold
```

plus timing bins:

```text
early / middle / late / none
```

First experiment sequence:

1. Audit event-mode labels.
2. Train event-mode probe.
3. Train event-conditioned cVAE with oracle event as upper bound.
4. Add predicted event-mode prior and top-M event mixture sampling.
5. Evaluate with the Gate 3 action head and Gate 3.0c usage audit.

## Key Rule

Separate upper bound from deployable route:

```text
oracle event mode = diagnostic upper bound
predicted event mode mixture = deployable model
```

## Promotion Criteria

Promote only if:

```text
predicted event-mode mixture beats Gate 3.0 default K=16
predicted event-mode mixture beats shuffled event-mode control
transition-window MSE improves
mean-replacement audit still shows sample-set usage
```

## Paper-Story Implication

If positive, the story becomes sharper:

```text
GeoMoCo-WM learns visually grounded future-motion modes over EEF geometry and
gripper-event timing, and downstream action heads can exploit those aligned
modes.
```
