# World-Model Rollout, cVAE Sample Readout, And Action-Head Multimodality

- Date: 2026-06-08
- Context: after Gate 2.4c cVAE stochasticity calibration and before Gate 2.4d
  sample readout.

## User Question

当前 cVAE 要解决的 sample 选择问题，和 world model 本身的多种 rollout 是什么关系？
后续 action head 还会不会也有多种 action 可选？

## Core Answer

cVAE sample selection and world-model multi-rollout are the same family of
problem, but they happen at different abstraction levels.

General world-model planning:

```text
context
  -> sample multiple future states / latent states / videos
  -> score or plan over rollouts
  -> choose an action
```

Current GeoMoCo-WM:

```text
vision + proprio + task
  -> GeoMoCo-cVAE
  -> sample multiple future EEF SE(3) motion trajectories
  -> score/read out one or aggregate several
  -> action decoder / action head
  -> action chunk
```

So the current `future_motion` samples are a minimal world-model rollout:

```text
not full future image/state rollout yet;
but already a multi-future rollout in motion space.
```

The sample readout problem is therefore not a detour. It is the smallest
version of the planning problem that every stochastic world model must solve:

```text
when the model predicts multiple possible futures, the controller must decide
which future is worth executing.
```

## Action-Head Multimodality

There are two possible routes.

### Stage A: Multimodal Future Motion, Deterministic Action Decoder

```text
sample K future_motion candidates
each future_motion -> deterministic action decoder
get K action chunks
scorer selects a future/action pair
```

This is the current preferred route because attribution is clean:

- diversity comes from the world-motion prior;
- the action decoder only tests whether a predicted future is executable;
- if performance improves, we can attribute the gain to future-motion sampling
  plus readout rather than to a stronger action generator.

### Stage B: Multimodal Future Motion, Multimodal Action Head

```text
sample K future_motion candidates
for each future_motion, sample M action chunks
score K x M future/action candidates
execute the selected or aggregated action
```

This is stronger, but introduces two uncertainty sources:

```text
future uncertainty
action-realization uncertainty
```

If introduced too early, it becomes hard to know whether gains come from
GeoMoCo future-motion modeling or from a powerful action head such as diffusion
policy / flow matching / MeanFlow.

## Mainline Staging

Recommended order:

```text
Stage A:
  cVAE future-motion samples
  deterministic action decoder
  lightweight sample scorer/readout

Stage B:
  add gripper/contact/executability scoring
  make the readout more like a planner

Stage C:
  upgrade action head to diffusion / flow / MeanFlow
  compare deterministic inverse dynamics vs multimodal action head

Stage D:
  full world-model planning:
    future-motion rollout
    future visual/state rollout
    action rollout
    value/success scoring
```

## Decision

Gate 2.4d should continue with lightweight ScoreNet first.

This is the next mainline step because Gate 2.4c already showed:

```text
good future-motion samples exist;
random sampling does not reliably choose them;
GT best-of-K is not deployable.
```

The first readout gate should therefore keep the action decoder deterministic
and only learn:

```text
ScoreNet(context, vision, future_motion_sample, decoded_action_sample)
  -> sample score
```

Only after this readout works should the project add a multimodal action head.
