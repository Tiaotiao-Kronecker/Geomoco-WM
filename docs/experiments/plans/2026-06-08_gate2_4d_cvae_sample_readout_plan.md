# Gate 2.4d cVAE Sample Readout Plan

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4d
- Purpose: convert the calibrated Gate 2.4c cVAE sample set into a deployable
  readout, instead of relying on GT-selected best-of-K.

## Motivation

Gate 2.4c showed that free-bits cVAE samples have useful coverage:

```text
prior-mean action MSE: 0.040931
sample-mean action MSE: 0.041199
best-of-K action MSE: 0.036894
```

The gap means:

```text
good samples exist, but random sampling does not select them reliably.
```

Therefore the next bottleneck is a sample scorer / readout.

## Relation To World-Model Rollout

This gate is not a side branch away from world modeling. It is the minimal
world-model planning problem in motion space.

General stochastic world-model control:

```text
context
  -> sample multiple future states / videos / latent rollouts
  -> score rollouts
  -> choose or aggregate an action
```

Current GeoMoCo-WM:

```text
vision + proprio + task
  -> GeoMoCo-cVAE
  -> sample multiple future EEF SE(3) motion rollouts
  -> score/read out one or aggregate several
  -> action decoder
  -> action chunk
```

The only difference is the rollout representation:

```text
full world model: future image/state/latent rollout
current gate: future EEF motion rollout
```

Therefore sample readout is the first planning layer for the world-motion
model, not a contradiction of the mainline.

## Related Work Position

This gate follows the candidate-generation-plus-readout family:

- BCQ: behavior VAE candidates plus Q-value selection.
- IBC: energy model over observation-action pairs.
- Trajectory Transformer: beam/search readout over candidate trajectories.
- Diffuser / QGPO: reward or energy-guided generative sampling.
- Visual MPC / PETS: candidate trajectories plus cost/reward scoring.

The first GeoMoCo-WM version should stay simpler than full Q-guided diffusion:

```text
freeze cVAE
sample K future-motion candidates
score each candidate with a lightweight learned head
select or soft-aggregate candidates
decode action through the frozen action decoder
```

## Dataset

Use the same formal slice as Gate 2.4c:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

Use the calibrated cVAE checkpoints:

```text
outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt
outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt
```

## First Scorer Design

Generate `K=16` samples for each validation window:

```text
future_motion_k = cVAE.sample(context, vision, z_k)
action_k = frozen_action_decoder(context, future_motion_k)
```

Train a scorer:

```text
score_k = ScoreNet(context, visual_grounding, future_motion_k, action_k)
```

Initial target options:

1. action-distance target:
   - target label is negative action MSE to the GT action chunk;
   - simplest offline ranking signal;
   - risk: learns the frozen decoder's metric, not real closed-loop success.
2. motion-distance target:
   - target label is negative future-motion MSE to GT;
   - cleaner geometry;
   - risk: less aligned with action value.
3. hybrid target:
   - weighted combination of action distance, motion distance, gripper/contact
     consistency diagnostics;
   - likely best once gripper/contact features are added.

For Gate 2.4d, start with action-distance ranking because the current bottleneck
is downstream action value.

## Action-Head Staging

Keep the first readout gate attribution-clean:

```text
Stage A:
  multimodal future-motion cVAE
  deterministic action decoder
  learned sample scorer/readout
```

Do not immediately add a multimodal action head. A diffusion / flow / MeanFlow
action head should be a later branch:

```text
Stage B:
  multimodal future-motion cVAE
  multimodal action head
  K future candidates x M action candidates
  scorer/value/executability readout
```

Reason: if future-motion sampling and action-head sampling are introduced at
the same time, it becomes unclear whether gains come from GeoMoCo-WM's
future-motion representation or from a stronger action generator.

Gate 2.4d should answer the narrower question first:

```text
given the calibrated cVAE samples,
can a lightweight scorer choose a better future/action pair than prior mean?
```

## Evaluation

Compare:

| readout | deployable? | purpose |
| --- | --- | --- |
| prior mean | yes | stable baseline |
| random sample mean | yes | naive stochastic baseline |
| oracle best-of-K action | no | upper-bound coverage diagnostic |
| learned scorer argmax | yes | main Gate 2.4d result |
| learned scorer soft aggregation | yes | lower-variance alternative |

Primary metrics:

- action MSE / MAE;
- action translation L2 in meters;
- SO(3) geodesic rotation in degrees;
- gripper MSE;
- future-motion MSE;
- scorer top-1 vs oracle-best rank;
- regret to oracle best-of-K:

```text
regret = action_mse(selected_sample) - action_mse(oracle_best_sample)
```

## Pass Criteria

Promote the scorer/readout if:

- learned scorer selected action MSE beats prior mean;
- learned scorer selected action MSE beats random sample mean;
- regret to oracle best-of-K is reduced;
- gains appear in both seeds, not only one;
- gripper does not regress badly.

## Stop Criteria

Do not promote if:

- learned scorer cannot beat prior mean;
- learned scorer simply learns sample norm or trivial prior-likelihood bias;
- action gains come with severe gripper/contact regression;
- top-1 rank is unstable across suites or seeds.

## Mainline Decision

If Gate 2.4d passes, GeoMoCo-WM can claim:

```text
visual-conditioned stochastic future-motion samples can be read out into
more action-useful candidates without GT best-of-K selection.
```

If it fails, keep using the cVAE prior mean as the deployable branch and move to
gripper/contact-aware modeling before stronger diffusion or Q-guided readouts.

## Mainline Position

Gate 2.4d is the immediate next mainline step.

If it passes, the next branch should add gripper/contact/executability scoring.
If that also passes, then it becomes reasonable to compare deterministic action
decoding against diffusion / flow / MeanFlow action heads.

## Execution Result

Gate 2.4d implemented the first lightweight deployable sample readout:

```text
SampleScoreNet(c_t, future_motion_k, decoded_action_k) -> score_k
target: action-distance ranking
K: 16
seeds: 7, 17
```

Mean result:

```text
prior mean action MSE: 0.040931
random sample action MSE: 0.041183
ScoreNet argmax action MSE: 0.040201
oracle best-of-K action MSE: 0.036895
```

Decision: promote ScoreNet as the first working deployable cVAE sample readout,
but treat it as a modest first step. It closes only `18.09%` of the prior-to-
oracle readout gap, so the next branch should add gripper/contact/executability
signals or stronger ranking objectives before multimodal action heads.

Formal records:

- `docs/experiments/runs/2026-06-08_gate2_4d_lightweight_cvae_sample_scorer.md`
- `docs/experiments/comparisons/2026-06-08_gate2_4d_sample_readout_vs_oracle.md`
