# GeoMoCo-WM As Multimodal Motion Prior And Next Mainline

- Date: 2026-06-10
- Context: Gate 2.5d/2.5e/2.6a readout bottleneck and positioning discussion

## Core Question

If GeoMoCo-WM can provide a multimodal motion prior whose sample set contains
good futures, does it need to select the future itself?

## Current Evidence

The current joint cVAE results show:

```text
prior mean action MSE:      around 0.0438
learned readout action MSE: around 0.0434
oracle best-of-K:           around 0.022
deterministic joint:        around 0.0407
```

This means:

```text
good future motions exist in the sample set
but small learned readouts do not reliably select them
```

Gate 2.5e and Gate 2.6a strengthen this diagnosis:

```text
event-aware scoring improves event metrics but hurts action MSE
temporal scoring slightly improves selected rank but not action MSE
```

Therefore, the bottleneck is not merely a missing scalar event weight or a
slightly stronger per-candidate encoder.

## First-Principles Positioning

Robot decision making separates into:

```text
prediction:
  what futures are plausible?

decision:
  which future/action should be selected under task and execution constraints?
```

A policy directly models:

```text
π(a | observation, task)
```

A motion prior models:

```text
p(future_motion | observation, task)
```

GeoMoCo-WM is more naturally the second object:

```text
visual/proprio/task
  -> multimodal future_delta_ee + future_gripper/event hypotheses
```

It should not be forced to solve the full decision problem internally unless it
has access to the relevant objective, contact constraints, success signal, and
execution cost.

## Why This Is More Like A World Model / Motion Prior

A world model is valuable because it provides structured futures that a planner,
value function, policy head, or search procedure can use. It does not need to
be an end-to-end action policy by itself.

GeoMoCo-WM's output space is:

```text
not pixels
not generic latent state
not direct action
but structured, low-dimensional, action-relevant future motion
```

This is a natural motion-prior interface:

```text
compact enough for downstream decision making
interpretable enough for phase/event diagnostics
multimodal enough to preserve plausible futures
```

## Implication For Readout

The previous readout route asked:

```text
cVAE samples K futures
ScoreNet selects one future
frozen action decoder outputs action
```

This is useful diagnostically, but it may be too strict as the main method:

```text
small ScoreNet must solve future selection
using only a proxy action-MSE target
```

The more natural downstream route is:

```text
cVAE samples K future-motion hypotheses
action head / planner consumes the whole future set
action head outputs action
```

This moves the selection or aggregation problem into the decision module.

## Concern About Predicting Action MSE

Directly predicting action MSE is a scalar task, but the target is noisy:

```text
demo action is only one valid behavior
MSE may penalize alternative valid futures
translation/rotation/gripper errors mix imperfectly
candidate quality depends on hidden task/execution context
```

Therefore, calibrated action-MSE prediction should not be treated as the only
next step.

## Concern About Sample-Level Action-Aware cVAE

Top-q sample-level action-aware cVAE can make cVAE samples more action-useful,
but it risks:

```text
overfitting to demonstration action
reducing multimodal diversity
collapsing samples toward a common mode
```

If used later, it must be a conservative ablation:

```text
K = 16
q = 4
small lambda
warmup
track diversity, KL, prior logvar, random sample metrics, event distribution
```

It should not replace the more basic question:

```text
can downstream action heads use the current multimodal motion prior?
```

## New Mainline Recommendation

Shift the mainline from scorer-only readout to:

```text
Gate 3.0:
motion-prior-conditioned action head / planner
```

Freeze the current joint cVAE first, and train a downstream action head that
consumes the set of sampled future motions:

```text
inputs:
  current context
  K sampled future_delta_gripper motions

output:
  action chunk
```

This directly tests the claim:

```text
GeoMoCo-WM provides useful multimodal future-motion priors
that improve downstream action prediction/planning
```

## Required Baselines

To keep attribution clean:

```text
1. direct context action head
2. context + prior mean future motion
3. context + K real cVAE samples
4. context + K shuffled-visual cVAE samples
5. context + GT future motion upper bound
```

Promotion requires:

```text
context + K real samples > direct context
context + K real samples > shuffled samples
ideally approach or beat prior-mean/frozen-decoder readout baselines
event metrics do not collapse
```

## Updated Positioning

The project should not overclaim:

```text
GeoMoCo-WM itself is the policy
```

The sharper claim is:

```text
GeoMoCo-WM learns a visually grounded, multimodal, structured future-motion prior.
This prior contains action-useful futures and can improve a downstream
action head or planner when consumed as a set of hypotheses.
```

