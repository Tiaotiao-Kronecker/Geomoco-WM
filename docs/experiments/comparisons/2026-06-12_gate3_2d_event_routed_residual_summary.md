# Gate 3.2d Event-Routed Residual Summary

## Question

Can explicit event-family routing repair the gripper open/close transition
bottleneck better than scalar transition weighting or a parallel gripper head?

## Result

Not enough. The route label is learnable, but the routed gripper residual is
not deployable.

| branch | overall MSE | gripper MSE | sustain MSE | transition MSE | transition gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.1f baseline | 0.034767 | 0.150052 | 0.022793 | 0.134087 | 0.827336 |
| Gate 3.2d base | 0.035624 | 0.154709 | 0.023131 | 0.138891 | 0.859218 |
| Gate 3.2d routed | 0.035774 | 0.155760 | 0.023358 | 0.138406 | 0.855824 |

Additional signal:

```text
route accuracy = 0.921017
```

## Interpretation

The model can classify the event family, and routing gives a tiny transition
gain:

```text
transition MSE: 0.138891 -> 0.138406
transition gripper MSE: 0.859218 -> 0.855824
```

But the gain is too small and comes with worse sustain and overall metrics.
This means the current residual does not solve the real bottleneck. The missing
piece is probably step-level timing inside the horizon, not only window-level
event family.

## Decision

Do not promote Gate 3.2d. Treat it as a useful negative ablation:

```text
event family is readable, but window-level gripper residual routing is too
coarse.
```

Next mainline:

```text
Gate 3.2e: step-wise gripper/event-timing head or temporal transition mask.
```

