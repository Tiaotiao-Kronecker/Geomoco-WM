# Gate 3.1a Event Mode Label Readiness

## Question

Are weak gripper-transition event modes stable enough to become the next
mode-structured prior target?

## Answer

Yes, with one caveat: mixed-transition modes are too rare for the main training
objective.

## Evidence

| check | result |
| --- | --- |
| total windows | 16518 |
| close transition windows | 878 |
| open transition windows | 807 |
| transition timing distribution | no step-0 shortcut; steps 0-7 range from 198 to 239 |
| train/val split | both splits contain all 10 observed modes |
| rare modes | only `mixed_transition::early` and `mixed_transition::middle` |

## Decision

Proceed to Gate 3.1b event-mode probe.

Use the main stable mode set:

```text
sustain_open::none
sustain_close::none
transition_close::{early,middle,late}
transition_open::{early,middle,late}
```

Keep mixed-transition labels in the artifact, but treat them as rare diagnostics
or merge them if a classifier needs a closed stable class set.

## Mainline Implication

Gate 3.1 can now test whether visual/context features predict explicit
gripper-event modes, and then whether an event-conditioned cVAE can allocate
sample diversity across those modes.
