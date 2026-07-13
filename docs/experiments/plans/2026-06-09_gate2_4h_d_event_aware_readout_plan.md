# Gate 2.4h-d Minimal Event-Aware Readout Plan

- Date: 2026-06-09
- Status: completed
- Position: after Gate 2.4h-c sample event-alignment analysis.

## Purpose

Gate 2.4h-c showed that the current cVAE has limited event-aligned sample
coverage, and that the flat action-MSE ScoreNet does not explicitly prefer
close/open transition timing. Gate 2.4h-d tests the smallest possible
event-aware readout change:

```text
flat action-rank target
  + weak event-alignment rank target
```

This is a diagnostic, not a promoted method.

## Design

Keep fixed:

```text
cVAE checkpoint
frozen action decoder
K=16 samples
ScoreNet architecture
episode-level split
action-MSE model selection
```

Add:

```text
event_target_weight = 0.1
event target = negative standardized transition-alignment error
```

The event-alignment error penalizes:

```text
wrong transition type
wrong close/open step
missing transition when GT has one
```

Promotion still requires deployable action MSE to improve or at least not
regress.

## Stop Criteria

Stop or mark as negative if:

- action MSE regresses from Gate 2.4d flat ScoreNet;
- event accuracy / macro-F1 / transition step metrics do not improve;
- any improvement is limited to a non-action metric and does not help readout.

