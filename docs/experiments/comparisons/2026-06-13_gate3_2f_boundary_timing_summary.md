# Gate 3.2f Boundary Timing Summary

## Question

Does supervising the exact close/open boundary start fix the transition-window
bottleneck?

## Result

Only partially. The step-routed branch improves transition windows inside the
Gate 3.2f run, but overall, gripper, and sustain metrics regress.

| branch | overall MSE | gripper MSE | sustain MSE | transition MSE | transition gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.1f reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 | 0.827336 |
| Gate 3.2f base | 0.035453 | 0.154943 | 0.023101 | 0.137941 | 0.851940 |
| Gate 3.2f step-routed | 0.035605 | 0.156003 | 0.023363 | 0.137172 | 0.846643 |

Additional signal:

```text
boundary-start accuracy = 0.986058
boundary positive fraction = 0.013576
no-boundary majority accuracy ~= 0.986424
```

The accuracy is therefore mostly a sparsity artifact. The real positive signal
is the small transition improvement:

```text
transition MSE: 0.137941 -> 0.137172
transition gripper MSE: 0.851940 -> 0.846643
```

## Interpretation

Boundary timing supervision is closer to the bottleneck than command-state
supervision, but the current unbalanced CE plus soft residual routing is still
too weak. It gives a targeted transition gain while damaging sustain windows
and the global action metric.

Across Gate 3.2b through Gate 3.2f, the pattern is now consistent:

```text
transition timing can be improved,
but simple deterministic heads trade away sustain/global quality.
```

## Decision

Do not promote Gate 3.2f.

Keep Gate 3.1f/Gate 3.1g full event/rank/prob top-4 as the deployable
reference:

```text
action MSE = 0.034767
gripper MSE = 0.150052
```

Next mainline:

```text
Gate 3.2g: boundary-quality audit plus transition-local repair.
```

The next branch should report boundary precision/recall or PR-AUC, not only
accuracy, and should try a transition-local or calibrated boundary gate before
moving to a heavier flow/diffusion action head.
