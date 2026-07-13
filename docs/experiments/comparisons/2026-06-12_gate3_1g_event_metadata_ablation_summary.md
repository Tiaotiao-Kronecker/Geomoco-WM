# Gate 3.1g Event Metadata Ablation Summary

## Question

Which event metadata channel explains the Gate 3.1f action-head gain?

## Result

The best result needs both:

```text
event identity
event rank/probability
```

| branch | action MSE | gripper MSE |
| --- | ---: | ---: |
| anonymous top-4 | 0.038024 | 0.167432 |
| event-only | 0.037108 | 0.163236 |
| rank/prob-only | 0.036069 | 0.158481 |
| shuffled-event/rank/prob | 0.036228 | 0.154256 |
| full event/rank/prob | 0.034767 | 0.150052 |

## First-Principles Interpretation

The future-motion set is not just a bag of trajectories. Each sample has a
latent reason for being proposed:

```text
which event mode it represents
how plausible that event mode is in the current context
where it sits in the event-prior ranking
```

If the action head sees only motion, it has to infer that structure again from
noisy samples. If it sees event identity without confidence, it knows the mode
but not whether the mode is context-compatible. If it sees confidence without
identity, it knows which rank is trusted but not what semantic manipulation
phase the sample represents.

The full metadata provides both axes and gives the best result.

## Decision

Keep the promoted interface as:

```text
future_delta_gripper sample
+ event-mode one-hot
+ event rank
+ event probability
```

Next, stress-test this interface before adding a flow/diffusion action head.
