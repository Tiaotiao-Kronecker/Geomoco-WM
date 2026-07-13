# Gate 3.2b Transition-Weighted Summary

## Question

Can we repair the Gate 3.2a transition-window failure by simply upweighting
transition windows in the action-head loss?

## Result

Partially, but not cleanly.

| branch | overall MSE | sustain MSE | transition MSE | transition gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f baseline | 0.034773 | 0.022793 | 0.134087 | 0.827336 |
| Gate 3.2b weight=2 | 0.035986 | 0.025233 | 0.125002 | 0.766794 |
| Gate 3.2b weight=4 | 0.037981 | 0.027829 | 0.122045 | 0.742795 |

## Interpretation

Transition weighting reduces transition error, especially gripper transition
error, which confirms that the failure mode is trainable. But it worsens
sustain and overall action MSE, so scalar weighting is too blunt.

The key lesson is:

```text
open/close timing should be modeled as a structured sub-problem,
not merely as a larger flat-MSE penalty.
```

## Decision

Keep Gate 3.1f full event/rank/prob top-4 as the default deployable interface.
Use Gate 3.2b as evidence for the next design: a transition-aware action head,
auxiliary future gripper/event timing head, or gated residual branch.
