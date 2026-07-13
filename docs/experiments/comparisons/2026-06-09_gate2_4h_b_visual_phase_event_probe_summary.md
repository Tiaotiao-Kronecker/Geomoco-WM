# Gate 2.4h-b Visual Phase/Event Probe Summary

- Date: 2026-06-09
- Status: completed
- Scope: determine whether visual grounding helps predict GeoMoCo
  phase/composition transition labels.

## Main Result

| branch | macro-F1 | interpretation |
| --- | ---: | --- |
| task_only | 0.215345 | weak shortcut baseline |
| task_proprio | 0.421542 | current robot state helps |
| future_motion_only | 0.442754 | future geometry contains event signal |
| proprio_future_motion | 0.509978 | geometry plus current state is stronger |
| visual_only | 0.630579 | DINO contains strong current phase evidence |
| visual_proprio_future_motion | 0.631385 | best overall |
| shuffled visual + proprio + future motion | 0.179985 | visual signal is alignment-sensitive |

## Decision

Visual grounding is useful for phase/event probing.

The most important control is shuffled visual:

```text
real visual + proprio + future motion macro-F1:     0.631385
shuffled visual + proprio + future motion macro-F1: 0.179985
```

This supports the main GeoMoCo-WM framing:

```text
visual context identifies manipulation phase;
future motion expresses candidate phase transitions;
event probes make the composition timing measurable.
```

## Mainline Implication

Move to Gate 2.4h-c before adding new scorer losses:

```text
cVAE sample event-alignment analysis
```

The next question is not whether GT future motion and visual context can
predict event labels. It is whether sampled GeoMoCo future motions and their
decoded actions contain event-aligned candidates that the current ScoreNet is
or is not selecting.
