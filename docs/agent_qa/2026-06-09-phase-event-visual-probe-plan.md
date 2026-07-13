# Phase/Event Probe And Visual Grounding Discussion

- Date: 2026-06-09
- Context: after Gate 2.4g hard-negative readout regressed from the flat
  ScoreNet baseline.

## Discussion Summary

The next readout issue should not be framed as a small gripper heuristic.
Gripper/contact event timing is a direct observable probe of GeoMoCo's intended
phase/composition advantage.

Core claim:

```text
GeoMoCo-WM should capture when manipulation phases transition:
  approach -> align -> close/contact -> transport -> release
```

Visual grounding can help because the same EEF motion can mean different
phases depending on object/scene state. DINO tokens should provide current
state evidence, while future-motion samples express candidate phase transitions.

## Mainline Plan

```text
Gate 2.4h-a:
  gripper/event label audit

Gate 2.4h-b:
  visual phase/event probe

Gate 2.4h-c:
  cVAE sample event-alignment analysis

Gate 2.4h-d:
  event-aware ScoreNet only if probes are positive
```

## Important Constraint

Event labels are derived from action/proprio streams, not from vision. Vision is
used as a predictor/input for phase understanding, not as a label generator.

The final method should not claim true contact labels unless force/tactile or
clean simulator contact state is available. Until then, contact is only a weak
proxy.
