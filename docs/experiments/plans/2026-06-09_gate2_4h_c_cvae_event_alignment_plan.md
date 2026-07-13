# Gate 2.4h-c cVAE Sample Event Alignment Plan

- Date: 2026-06-09
- Status: completed
- Position: after Gate 2.4h-b visual phase/event probe, before training any
  event-aware ScoreNet.

## Purpose

Gate 2.4h-b showed that aligned DINO visual features strongly predict
gripper-transition phase labels. Gate 2.4h-c asks the next mainline question:

```text
Do visual cVAE future-motion samples contain close/open transition candidates,
and do existing readouts select those event-aligned samples?
```

This gate is diagnostic only. It does not train a new scorer.

## Method

For each validation window:

```text
cVAE prior mean / K prior samples
  -> frozen action decoder
  -> decoded action chunk
  -> transition event label
```

Then compare the decoded event label to the GT transition label from
Gate 2.4h-a.

## Readouts

```text
prior_mean
random_sample_mean
event_oracle_best          # non-deployable diagnostic
flat_action_oracle_best    # non-deployable action upper-bound selector
se3_gripper_oracle_best    # non-deployable structured selector
scorer_argmax              # deployable Gate 2.4d ScoreNet selector
```

## Metrics

- event type accuracy / macro-F1;
- transition type accuracy on close/open/mixed transition windows;
- transition step exact / within-1 / within-2;
- sample coverage: whether any of K samples contains the correct event type or
  timing;
- selected event-oracle rank for flat oracle, structured oracle, and ScoreNet.

## Pass Signal

Pass as a diagnostic if:

- event oracle best-of-K improves over prior mean;
- any-sample coverage is meaningfully above prior mean;
- ScoreNet-selected samples rank near event oracle, or at least do not destroy
  transition alignment.

If samples do not contain event-aligned candidates, do not train event-aware
ScoreNet. If samples contain them but ScoreNet does not select them, then the
next bottleneck is event/executability-aware readout.

