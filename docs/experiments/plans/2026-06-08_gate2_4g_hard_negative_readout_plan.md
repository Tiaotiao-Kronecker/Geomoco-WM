# Gate 2.4g Hard-Negative Readout Plan

- Date: 2026-06-08
- Status: completed
- Position in mainline: after Gate 2.4f structured oracle evaluation.

## Purpose

Gate 2.4f showed that naive `SE(3)` / `SE(3)+gripper` metric-target
replacement improves structured oracle rank but does not improve deployable
action MSE. Gate 2.4g tests whether a lightweight hard-negative auxiliary loss
can teach the scorer to reject samples that are geometrically plausible but
decode to worse actions.

## Minimal Mechanism

The base scorer loss remains the Gate 2.4d listwise flat-action ranking loss:

```text
positive readout signal = lower MSE(action_k, action_gt)
```

The auxiliary hard-negative loss adds a pairwise constraint:

```text
positive sample = flat action oracle sample
negative sample = best SE(3)+gripper oracle sample, excluding the flat oracle
loss = softplus(logit_negative - logit_positive)
```

Training objective:

```text
loss = listwise_action_loss + w_hardneg * hard_negative_loss
```

This is intentionally conservative: it does not change the scorer architecture,
cVAE, action decoder, dataset, K, or promotion metric.

## Tested Weights

| branch | hard negative target | weight | margin |
| --- | --- | ---: | ---: |
| smoke | `se3_gripper` | 0.5 | 0.0 |
| formal | `se3_gripper` | 0.1 | 0.0 |
| formal | `se3_gripper` | 0.5 | 0.0 |

## Promotion Criteria

Promote only if the branch improves over Gate 2.4d flat ScoreNet on:

- deployable action MSE;
- flat oracle rank or top-1 match;
- no severe regression in `SE(3)` / gripper diagnostics.

If it does not improve, treat it as a negative ablation and move to explicit
executability/contact proxy construction.
