# Gate 2.4e SE(3)+Gripper-Aware Sample Scorer Plan

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4e
- Purpose: improve the Gate 2.4d lightweight cVAE sample readout by replacing
  flat action-MSE ranking with structured SE(3)+gripper-aware ranking targets.

## Motivation

Gate 2.4d showed that a deployable ScoreNet can select better cVAE samples than
prior mean or random sampling:

```text
prior mean action MSE: 0.040931
random sample action MSE: 0.041183
ScoreNet argmax action MSE: 0.040201
oracle best-of-K action MSE: 0.036895
```

But the readout remains weak:

```text
prior-to-oracle readout gap closed: 18.09%
top1 oracle match: 0.238
selected oracle rank among K=16: 6.61
```

The likely issue is that the 2.4d target is still a flat action-MSE target.
It mixes translation, rotation coordinates, and gripper into one Euclidean
number.

## Question

Can a structured scorer target better identify action-useful samples?

Specifically, compare:

```text
2.4d flat action-MSE target
2.4e-a SE(3)-geodesic target
2.4e-b SE(3)-geodesic + gripper target
```

## Target Design

For each sampled future-motion candidate:

```text
future_motion_k = cVAE.sample(c_t, z_k)
action_k = frozen_action_decoder(context, future_motion_k)
```

Compute candidate errors:

```text
translation_m_l2_k
rotation_geodesic_rad_k
gripper_mse_k
```

Then train ScoreNet with a listwise ranking target:

```text
score target_k = - weighted_structured_error_k
loss = CE(softmax(target_k), log_softmax(score_k))
```

Initial variants:

```text
se3:
  error = w_trans * zscore(translation_m_l2)
        + w_rot   * zscore(rotation_geodesic_rad)

se3_gripper:
  error = w_trans * zscore(translation_m_l2)
        + w_rot   * zscore(rotation_geodesic_rad)
        + w_grip  * zscore(gripper_mse)
```

Default weights:

```text
w_trans = 1.0
w_rot = 1.0
w_grip = 1.0
```

## Dataset And Checkpoints

Use the same two-files-per-suite slice:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

Use Gate 2.4c cVAE checkpoints:

```text
outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt
outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt
```

Use frozen Gate 1.6 action decoders:

```text
outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Evaluation

Compare against Gate 2.4d:

| branch | deployable? | purpose |
| --- | --- | --- |
| prior mean | yes | cVAE stable baseline |
| random sample mean | yes | naive stochastic baseline |
| 2.4d flat ScoreNet | yes | current readout baseline |
| 2.4e SE(3) ScoreNet | yes | structured geometric readout |
| 2.4e SE(3)+gripper ScoreNet | yes | structured executable readout |
| oracle best-of-K action | no | GT-selected upper-bound diagnostic |

Primary metrics:

- action MSE;
- action translation L2 in meters;
- SO(3) geodesic rotation in degrees;
- gripper MSE;
- top1 oracle match;
- selected oracle rank;
- regret to oracle best-of-K.

## Pass Criteria

Promote if the structured scorer:

- beats Gate 2.4d flat ScoreNet on mean action MSE, or
- clearly improves gripper MSE without regressing action MSE, or
- improves oracle-rank / regret enough to justify the structured target.

## Stop Criteria

Do not promote if:

- action MSE regresses relative to Gate 2.4d;
- SE(3) metrics improve but gripper collapses;
- gripper improves only by worsening translation/rotation substantially;
- gains appear in only one seed.

## Mainline Position

Gate 2.4e should run before multimodal action heads. It tests whether the
motion-rollout readout can be improved while keeping the action decoder frozen
and deterministic.

## Completion Summary

Run record:

```text
docs/experiments/runs/2026-06-08_gate2_4e_se3_gripper_aware_sample_scorer.md
```

Comparison:

```text
docs/experiments/comparisons/2026-06-08_gate2_4e_structured_readout_vs_flat.md
```

Mean action MSE:

```text
Gate 2.4d flat action-MSE target: 0.040201
Gate 2.4e SE(3) target:          0.040582
Gate 2.4e SE(3)+gripper target:  0.040424
prior mean baseline:             0.040931
oracle best-of-K diagnostic:     0.036895
```

Decision: do not promote the naive structured target replacement. It improves
over the prior mean but regresses from the Gate 2.4d flat action-MSE scorer.
The next readout branch should use hard-negative ranking or richer
gripper/contact/executability proxies rather than only changing metric units.
