# Gate 2.4e Structured Readout Vs Flat Action-MSE Readout

- Date: 2026-06-08
- Status: completed
- Scope: compare the Gate 2.4d flat action-MSE ScoreNet target against
  Gate 2.4e SE(3) and SE(3)+gripper structured ScoreNet targets.

## Summary

| branch | deployable? | action MSE | gripper MSE | top1 oracle | oracle rank | gap closed | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gate 2.4c prior mean | yes | 0.040931 | 0.167670 | - | - | 0.00% | stable cVAE baseline |
| Gate 2.4d flat ScoreNet | yes | 0.040201 | 0.165270 | 0.238 | 6.614 | 18.09% | current best deployable readout |
| Gate 2.4e SE(3) ScoreNet | yes | 0.040582 | 0.167254 | 0.179 | 7.525 | 8.64% | improves over prior but regresses from flat target |
| Gate 2.4e SE(3)+gripper ScoreNet | yes | 0.040424 | 0.166138 | 0.186 | 7.239 | 12.57% | better than pure SE(3), still worse than flat target |
| Oracle best-of-K action | no | 0.036895 | 0.152010 | 1.000 | 1.000 | 100.00% | GT-selected diagnostic upper bound |

## What Changed

Gate 2.4d trained the scorer with:

```text
target_k = - standardized flat MSE(action_k, gt_action_chunk)
```

Gate 2.4e tested:

```text
se3:
  target_k = - standardized translation_m_l2
           + - standardized rotation_geodesic_rad

se3_gripper:
  target_k = se3 target
           + - standardized gripper_mse
```

The scorer input, frozen cVAE, frozen action decoder, dataset slice, K, epochs,
batch size, and seeds remained aligned with Gate 2.4d.

## Interpretation

The structured targets are useful diagnostically but are not better training
targets for the current lightweight ScoreNet.

The likely reason is that flat action MSE, although less physically
interpretable, is closer to the downstream decoder objective. The structured
target splits the same decoded action into separate terms, but it does not add
new information about contact timing, grasp success, object progress, or
execution feasibility.

The gripper term helps relative to pure SE(3), but only modestly:

```text
SE(3) action MSE:          0.040582
SE(3)+gripper action MSE:  0.040424
flat action-MSE action MSE:0.040201
```

This suggests the next readout problem is not just metric units. The scorer
needs richer sample-difficulty supervision.

## Decision

Keep Gate 2.4d flat ScoreNet as the current readout baseline.

Do not promote Gate 2.4e structured target replacement.

Next mainline readout branch:

1. construct hard negatives among cVAE samples that look close in SE(3) but
   decode to worse actions;
2. add gripper/contact/executability proxies only when they add information
   beyond decoded action error;
3. consider a scorer objective that predicts downstream regret or rank directly
   instead of a hand-weighted metric sum.

This remains before multimodal action heads in the mainline, because the
motion-rollout readout still has unused oracle best-of-K headroom.
