# PointWorld vs GeoMoCo-WM: Multimodal Rollouts

- Date: 2026-06-10
- Context: after Gate 2.5c joint GeoMoCo-cVAE and before Gate 2.5d joint
  cVAE sample readout/scorer.

## User Question

PointWorld also seems to produce multiple rollouts. On the specific issue of
`multimodal`, how is that different from GeoMoCo-WM?

## Short Answer

The difference is where the diversity lives.

PointWorld-style multiple rollouts are best understood as planner-side action
candidate rollouts:

```text
observation
  -> propose K candidate action sequences
  -> forward world model predicts one future response for each action sequence
  -> planner scores predicted futures with a task cost
  -> execute the first action of the best sequence
```

GeoMoCo-WM's current multimodality is model-intrinsic future-motion sampling:

```text
vision + proprio + task
  -> GeoMoCo-cVAE prior
  -> sample K future-motion hypotheses
  -> scorer/readout selects or aggregates future-motion samples
  -> deterministic action decoder maps the selected future to action
```

So:

```text
PointWorld:
  action diversity -> multiple predicted world rollouts

GeoMoCo-WM:
  latent future-motion diversity -> multiple candidate futures
```

This distinction matters because the two systems solve different planning
subproblems.

## PointWorld Multimodality

Based on the public PointWorld paper/project description, PointWorld is a
3D world model that conditions on RGB-D observations and low-level robot action
commands, and predicts 3D point-flow / scene response in a shared 3D geometric
space.

Its multimodality is therefore mainly about:

- sensor and state/action fusion: RGB-D observations plus robot action commands;
- geometric unification: representing state change as 3D point flow;
- embodiment/control compatibility: using the same geometric prediction
  interface for planning/control.

If multiple futures appear during control, the most likely interpretation is
MPC/search:

```text
sample or optimize many action sequences
roll each one forward through the world model
score the resulting predicted world states
choose the best action sequence
```

That is different from a latent generative model producing multiple futures for
the same context before any action sequence is chosen.

Important caveat: this note distinguishes the high-level design implied by the
public paper/site. If the released implementation contains an additional
stochastic latent or diffusion sampler, this distinction should be revisited.

## GeoMoCo-WM Multimodality

The current GeoMoCo-WM mainline uses multimodality for a narrower and more
action-interface-centered question:

```text
Given current vision/proprio/task context,
what future motion structures could be useful for action?
```

After Gate 2.5c, the promoted future representation is:

```text
future_delta_ee + future_gripper/event
```

The model is not trying to generate dense future images or full point-cloud
scene dynamics. Instead, it samples structured future-motion hypotheses that
are meant to be tested through a controlled motion-to-action bridge.

The current evidence is:

```text
deterministic joint baseline action MSE: 0.040688
cVAE prior mean action MSE:             0.043816
cVAE best-of-K action MSE:              0.022139
```

This means good samples exist in the joint future-motion sample set, but the
project still needs a deployable readout/scorer to select them without GT.

## Narrative Implication

GeoMoCo-WM should not position itself as a PointWorld-style dense 3D world
model. The sharper claim is:

```text
GeoMoCo-WM is a visual-grounded, geometry-structured, multimodal
future-motion interface for action.
```

PointWorld's useful lesson is that multiple rollouts need a planner/readout.
But GeoMoCo-WM should keep the source of diversity explicit:

```text
diversity comes from future-motion latent samples first,
not from sampling many action sequences first.
```

This supports the current Gate 2.5d sequencing:

```text
freeze / reuse the joint cVAE sample generator
train a lightweight scorer/readout over future_delta_gripper samples
compare against deterministic joint, prior mean, and oracle best-of-K
```

Only after this readout is interpretable should the project consider a
PointWorld-like outer MPC layer that samples action sequences directly, or a
multimodal diffusion/flow action head.

## Sources

- PointWorld paper: <https://arxiv.org/abs/2601.03782>
- PointWorld project page: <https://point-world.github.io/>
- Local mainline record:
  `docs/worklog/active.md`
- Local sample-readout note:
  `docs/agent_qa/2026-06-08-world-model-rollout-action-head-and-sample-readout.md`
