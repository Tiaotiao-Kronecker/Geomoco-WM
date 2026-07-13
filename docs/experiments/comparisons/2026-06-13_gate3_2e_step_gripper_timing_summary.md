# Gate 3.2e Step Gripper Timing Summary

## Question

Does per-step gripper command-state supervision fix the transition bottleneck?

## Result

No. The command-state classifier is accurate, but the step-routed action output
is worse than the base output.

| branch | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f baseline | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.2e base | 0.035254 | 0.153039 | 0.022961 | 0.137397 |
| Gate 3.2e step-routed | 0.035397 | 0.154042 | 0.023084 | 0.137707 |

Additional signal:

```text
gripper step command accuracy = 0.947207
```

## Interpretation

This branch answered an important first-principles question: the missing signal
is not simply per-step open/close command state. That label is too easy because
LIBERO gripper commands are usually explicit at every step.

The bottleneck is transition-boundary timing:

```text
when does the gripper state change?
```

not command-state classification:

```text
what command sign is emitted at this step?
```

## Decision

Do not promote Gate 3.2e.

Next mainline:

```text
Gate 3.2f: step-wise transition-boundary timing using close_step/open_step
from the event-mode audit JSON.
```

