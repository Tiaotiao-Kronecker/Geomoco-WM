# GeoMoCo-WM Positioning Advantages vs Related Work

- Date: 2026-06-10
- Context: after clarifying PointWorld-style action-candidate rollouts vs
  GeoMoCo-WM latent future-motion rollouts, and before Gate 2.5d joint cVAE
  sample readout/scorer.

## User Question

Compared with nearby work such as PointWorld, AMPLIFY, PlaNet/Dreamer, SV2P,
Diffuser, ACT, BCQ, and IBC, what is the potential advantage of the current
GeoMoCo-WM positioning?

## Short Answer

The novelty should not be framed as:

```text
latent can generate multiple possible futures
```

That idea already exists in stochastic video prediction, latent world models,
diffusion planners, CVAE action policies, and offline-RL candidate generators.

The sharper GeoMoCo-WM positioning is:

```text
Put stochastic future rollout in an action-relevant geometric motion space,
instead of pixels, generic latent states, or direct action chunks.
```

The current interface is:

```text
vision + proprio + task
  -> multimodal future_delta_ee + future_gripper/event samples
  -> scorer/readout
  -> deterministic action decoder
```

## Potential Advantages

### 1. More Action-Relevant Than Pixel Or Dense 3D Prediction

Video prediction and dense 3D world models are expressive but expensive. They
predict many visual/geometric details that may not affect the next action.

GeoMoCo-WM instead predicts a narrower interface:

```text
EEF geometric future + gripper/event timing
```

This is less complete than dense scene prediction, but it is closer to the
manipulation control interface and therefore may need fewer samples and less
model capacity to affect action value.

### 2. More Interpretable Than Generic Latent World Models

PlaNet/Dreamer-style latent imagination is powerful, but failures inside the
latent space are hard to diagnose.

GeoMoCo-WM keeps the rollout space inspectable:

```text
translation error
rotation / SO(3) geodesic error
gripper MSE
close/open transition timing
decoded action value
```

This makes it easier to tell whether a failure comes from visual grounding,
geometry, event timing, readout, or the motion-to-action bridge.

### 3. Cleaner Attribution Than Direct Strong Action Policies

ACT, Diffusion Policy, flow matching, and MeanFlow-style action heads can be
strong, but they obscure whether improvements come from the action generator or
from the future-motion representation.

GeoMoCo-WM currently keeps the bridge controlled:

```text
future-motion prior -> deterministic action decoder
```

This lets the project test whether predicted future motion itself has
downstream action value before upgrading the action head.

### 4. Explicit Gripper/Event Semantics

Old GeoMoCo evidence suggested that EEF-only motion is too weak as a policy
interface. Gate 2.4i and Gate 2.5 show that the missing channel is largely
gripper/open-close event timing.

The current joint representation:

```text
future_delta_ee + future_gripper/event
```

is therefore not just a trajectory latent. It is a compact manipulation-phase
interface that can express reach, contact, close, transport, open/release, and
recover-like structure.

### 5. Structured Candidate Space For Readout

PointWorld-style MPC samples candidate action sequences and rolls each one
forward. GeoMoCo-WM samples candidate future-motion structures first, then
decodes action.

```text
PointWorld-style:
  action sequence candidates -> world rollout -> task-cost score

GeoMoCo-WM:
  future-motion candidates -> action bridge -> action-value score
```

If the future-motion space is learned well, readout may be more sample-efficient
and interpretable than raw action-sequence search.

## What Should Be Claimed Carefully

Current evidence supports the following cautious claim:

```text
The joint cVAE sample set contains useful future-motion candidates.
```

The evidence is:

```text
deterministic joint baseline action MSE: 0.040688
cVAE prior mean action MSE:             0.043816
cVAE best-of-K action MSE:              0.022139
```

This is not yet a deployable policy claim because best-of-K uses GT selection.
Gate 2.5d must show that a scorer/readout can recover part of this gap without
GT.

## Mainline Implication

Gate 2.5d should be evaluated as the first deployable test of the above
positioning:

```text
Can a learned readout select action-useful future_delta_ee + future_gripper/event
samples from the joint cVAE?
```

If it works, the project can claim a meaningful advantage over prior-mean CVAE
usage and can then consider richer scorers, contact/executability proxies,
outer MPC, or multimodal action heads.

If it fails, the current stochastic future-motion interface remains only a
coverage diagnostic, and the next fix should target readout supervision,
contact/event modeling, or the action decoder rather than claiming multimodal
world-model value.

## Sources

- PointWorld: <https://arxiv.org/abs/2601.03782>
- AMPLIFY: <https://arxiv.org/abs/2506.14198>
- PlaNet: <https://arxiv.org/abs/1811.04551>
- Dreamer: <https://arxiv.org/abs/1912.01603>
- SV2P: <https://arxiv.org/abs/1710.11252>
- Diffuser: <https://arxiv.org/abs/2205.09991>
- ACT: <https://arxiv.org/abs/2304.13705>
- BCQ: <https://arxiv.org/abs/1812.02900>
- IBC: <https://arxiv.org/abs/2109.00137>
- Local mainline record: `docs/worklog/active.md`
