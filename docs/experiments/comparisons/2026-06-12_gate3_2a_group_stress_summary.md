# Gate 3.2a Group Stress Summary

## Question

Does the promoted event-aware top-4 action-head interface remain reliable when
we inspect meaningful subgroups rather than only the global mean?

## Result

The global mean is stable:

| branch | action MSE | gripper MSE |
| --- | ---: | ---: |
| Gate 3.1f full event/rank/prob top-4 | 0.034767 | 0.150052 |
| Gate 3.2a group-audit overall | 0.034773 | 0.150159 |

But the error is highly concentrated in transition windows:

| group | action MSE | gripper MSE |
| --- | ---: | ---: |
| sustain windows | 0.022793 | 0.068512 |
| transition windows | 0.134087 | 0.827336 |
| transition open | 0.150220 | 0.898143 |
| transition close | 0.118580 | 0.758840 |

## Interpretation

The current interface is not mainly failing on generic SE(3) geometry.
Transition windows have only a small translation penalty and do not have worse
rotation geodesic error than sustain windows. Their action loss is dominated by
gripper/event timing.

This sharpens the mainline:

```text
GeoMoCo-WM's next bottleneck is precise open/close timing under event-structured
sample consumption.
```

## Decision

Keep event-aware top-4 as the current best interface, but do not move to
flow/diffusion action heads yet. The next mainline should run Gate 3.2b:
transition-focused action-head stress, preferably with transition-balanced
training, transition-timing auxiliary supervision, or an oracle-transition
metadata upper bound.
