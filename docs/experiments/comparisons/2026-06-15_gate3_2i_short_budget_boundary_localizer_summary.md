# Gate 3.2i Short-Budget Boundary Localizer Summary

## Question

Can a window-level close/open step-index localizer recover the Gate 3.2h oracle
boundary-mask gain with a small implementation and run budget?

## Result

No.

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.2h oracle boundary mask | 0.032018 | 0.133072 | 0.021841 | 0.116469 |
| Gate 3.2h best predicted mask | 0.035201 | 0.155339 | 0.021902 | 0.145723 |
| Gate 3.2i boundary-index readout | 0.035053 | 0.152685 | 0.022868 | 0.135682 |

Within the same 3.2i checkpoints, the boundary-index readout is worse than the
base action output:

```text
3.2i base mean MSE           = 0.034878
3.2i boundary-index mean MSE = 0.035053
```

## Interpretation

The boundary-index objective avoids the extreme sparse-CE failure mode, but it
still does not localize close/open steps well enough. Exact event-window
accuracy is only `0.069527` for close and `0.023539` for open; within-1 is
`0.249839` and `0.160918`.

This is enough to slightly improve over Gate 3.2h's best predicted mask, but
not enough to beat the Gate 3.1f/Gate 3.1g deployable reference or approach the
oracle-mask upper bound.

## Decision

Do not promote Gate 3.2i.

Option A's short-budget deterministic boundary-localization branch is now
exhausted. The next mainline should pivot to a richer temporal/flow action
decoder that models the transition gripper trajectory directly.
