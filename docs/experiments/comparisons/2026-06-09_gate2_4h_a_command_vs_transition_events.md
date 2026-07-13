# Gate 2.4h-a Command-State Vs Transition Event Labels

- Date: 2026-06-09
- Status: completed
- Scope: choose the event-label contract for later visual phase/event probes.

## Comparison

| label mode | useful for phase timing? | close event windows | open event windows | shortcut risk | decision |
| --- | --- | ---: | ---: | --- | --- |
| command-state | weak | 7,481 including mixed | 10,534 including mixed | high, close step mostly 0 | do not use as main phase label |
| transition | yes | 892 including mixed | 821 including mixed | none detected by current rule | use for Gate 2.4h-b/c/d |

## Why Transition Wins

Command-state labels answer:

```text
is the gripper command currently close/open inside this future chunk?
```

That is not the phase boundary. Most chunks begin already in a sustained
open/close state, so the first close/open step collapses to step 0.

Transition labels answer:

```text
does this future chunk enter close/open from the previous command state?
```

That is the phase boundary signal we want for GeoMoCo:

```text
approach/alignment -> close/contact
transport -> release
```

## Mainline Implication

The next visual probe should predict transition labels, not command-state
labels. Sustained states are still useful as context classes, but the main
timing metric should be close/open transition timing.

This reduces overfitting risk because a model cannot pass the probe merely by
predicting that the first future step has the same command state as the current
policy segment.
