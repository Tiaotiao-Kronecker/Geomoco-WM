# Cross-Paper Lessons For GeoMoCo-WM

- Date: 2026-06-08
- Project: `Geomoco-WM`
- Status: design synthesis
- Context: after the OASIS, ZipMotion / ZipMo, AMPLIFY, Fast-WAM, Mean Flow
  Policy / MVP, SE(3)-equivariant Spherical Diffusion Policy, GuidedVLA, and
  current Gate 2.4c / 2.4d GeoMoCo-WM discussions.

## Purpose

This note summarizes what the recent paper analysis changes for GeoMoCo-WM.
It is not a generic related-work list. The goal is to preserve the design
decisions that matter for the private project:

```text
What should GeoMoCo-WM be?
What should it not become?
Which paper lessons should become immediate gates, baselines, or future branches?
```

## Current Local State

GeoMoCo-WM has moved beyond the initial blueprint.

Local gates now show:

- ground-truth future EEF motion strongly improves the controlled action
  decoder over direct context;
- visual future-motion priors now beat direct context through a frozen decoder;
- action-aware visual prior training improves the learned branch;
- the first cVAE was nearly deterministic;
- free-bits calibration made the cVAE latent active and improved best-of-K
  action value;
- random cVAE samples still do not beat the prior mean, so the next bottleneck
  is sample scorer / readout.

Therefore the immediate mainline is:

```text
vision + proprio + task
  -> calibrated GeoMoCo-cVAE future-motion samples
  -> sample scorer / readout
  -> deterministic action decoder
  -> action chunk
```

Do not jump directly to a stronger action head until this readout gate is
understood.

## Core Decision

GeoMoCo-WM should be framed as:

```text
a visual-grounded stochastic world-motion prior
with phase/progress/composition structure
and a controlled motion-to-action interface.
```

It should not be framed as:

```text
only an EEF trajectory policy
only a VLA finetuning recipe
only a video world model
only a stronger action decoder
only a VAE compression trick
```

The central claim should stay representation/planning-facing:

```text
structured future-motion samples can provide action-useful rollouts
when they are visually grounded, scored/read out, and decoded by a learned executor.
```

## Paper Lesson Matrix

| Work | What It Really Teaches | What GeoMoCo-WM Should Borrow | What To Avoid |
| --- | --- | --- | --- |
| Fast-WAM / Wan | Training-time world modeling can improve action without test-time future video generation. | Keep action-facing evaluation, compare against direct action branches, and do not require future RGB rollout for the first claim. | Do not let a huge action head hide whether the future-motion representation matters. |
| ZipMotion / ZipMo | Long-horizon visual motion can be compressed and generated more efficiently than RGB video. | Treat future motion as the right abstraction level; continuous latents are a strong reference for smooth motion. | ZipMo tracks are not `SE(3)` composition; policy value still depends on a learned action head. |
| AMPLIFY | Motion prior and inverse dynamics should be separated, especially when action labels are scarce. | Keep `future motion -> learned action decoder` as a first-class interface; consider inverse-dynamics executability as a scorer signal. | Do not assume discrete FSQ tokens are better for GeoMoCo's continuous `SE(3)` geometry. |
| OASIS | EEF `SE(3)` trajectory supervision helps only when pose-supervised states condition a learned decoder. | Keep the oracle future-motion gate and learned decoder; use OASIS as the closest EEF-trajectory policy baseline. | Do not claim novelty if GeoMoCo becomes only `future EEF trajectory -> action decoder`. |
| SE(3)-equivariant SDP | Action policies benefit from respecting `SE(3)` geometry, especially for pose generalization and data efficiency. | Add a geometry-aware policy/action-head branch later, and use pose-OOD tasks to test it. | Do not confuse policy-side equivariance with world-model phase/composition. |
| GuidedVLA | Action decoders can be improved by assigning heads to object, depth, and skill factors. | Inject GeoMoCo phase/progress/composition as guided decoder factors or a ControlNet-style branch. | Do not expect a monolithic action decoder to discover every useful factor implicitly. |
| Mean Flow Policy / MVP | One-step action generation can be made fast with a mean-velocity constraint. | Use MeanFlow-style heads as later controlled action-decoder baselines. | Do not introduce a powerful action generator before the future-motion prior/readout value is isolated. |
| SE3-Nets / structured rigid-motion line | Rigid object/part motion should be modeled through shared transforms rather than unconstrained dense flow. | Keep `SE(3)` composition and geodesic metrics central. | Do not make full object-relative `SE(3)` a first blocker when labels are expensive. |

## Design Implications

### 1. Keep The Motion-To-Action Interface, But Do Not Hardcode It

OASIS and AMPLIFY point to the same first-principles boundary:

```text
motion is not action;
motion needs a learned inverse-dynamics / action decoder.
```

For GeoMoCo-WM, the executor should remain learned:

```text
context + future_motion -> action chunk
```

Hardcoding:

```text
EEF(t+1) - EEF(t) -> action
```

is too brittle because it ignores controller conventions, normalization,
gripper timing, contact, prediction noise, and dataset-specific action
semantics.

### 2. Treat Current cVAE Samples As Minimal World-Model Rollouts

Gate 2.4c means the cVAE is no longer just a reconstruction module. It is now a
short-horizon stochastic rollout model in motion space:

```text
context -> K possible future EEF motion trajectories
```

This is the smallest version of world-model planning:

```text
sample futures -> score futures -> choose action
```

Therefore Gate 2.4d sample readout is not a side quest. It is the first
planning layer.

### 3. Use Prior Mean, Random Samples, Best-Of-K, And Scorer As Different Claims

These should never be mixed:

| Readout | Deployable? | Claim |
| --- | --- | --- |
| prior mean | yes | stable deterministic cVAE policy interface |
| random sample mean | yes | naive stochastic deployment |
| best-of-K | no | oracle coverage diagnostic |
| learned scorer argmax | yes | deployable sample readout |
| learned scorer soft aggregation | yes | lower-variance readout |

If best-of-K is strong but learned readout fails, the cVAE has coverage but not
deployable planning value yet.

### 4. Keep Phase / Progress / Composition Separate

GuidedVLA's skill head and the old `u_t` discussion reinforce the same rule:

```text
phase, progress, and composition should not be collapsed into one scalar.
```

Recommended roles:

```text
u_geom:
  normalized EEF geometric motion progress;
  cheap scaffold for windows, phase bins, and probes.

phase / skill:
  stage or mode of the current motion;
  can use temporal rank, skill labels, gripper/contact landmarks.

composition:
  algebraic relation between motion segments;
  evaluate with `SE(3)` geodesic closure and latent closure.

semantic progress:
  task-specific object/contact/success progress;
  use only as diagnostic or upper bound unless labels are clean.
```

### 5. Use Weak Object Grounding Before Expensive Object `SE(3)`

The OASIS / GuidedVLA contrast is important.

OASIS shows EEF `SE(3)` is practical because demonstrations already contain EEF
poses.

GuidedVLA shows object masks and depth features can improve action without
requiring full object `SE(3)` tracking.

For GeoMoCo-WM:

```text
do not make robot-object relative SE(3) the next blocker.
```

Instead, use cheaper grounding first:

- DINO patch tokens;
- depth features;
- task-relevant object masks when available;
- gripper/contact landmarks;
- optional object-state diagnostics only when clean.

### 6. Delay Strong Multimodal Action Heads

Mean Flow Policy, Diffusion Policy, SDP, and related action generators are
strong future branches. But adding them too early destroys attribution:

```text
future-motion prior value
vs.
strong action-head value
```

Recommended staging:

```text
Stage A:
  cVAE future-motion samples
  deterministic action decoder
  lightweight scorer/readout

Stage B:
  add gripper/contact/executability scoring
  keep action decoder controlled

Stage C:
  upgrade action head to diffusion / flow / MeanFlow / SDP-style geometry-aware policy

Stage D:
  combine multimodal future-motion samples with multimodal action sampling
```

## Immediate Mainline

### Gate 2.4d: Sample Scorer / Readout

Implement the minimal deployable readout:

```text
freeze calibrated cVAE
sample K future-motion candidates
decode each through frozen action decoder
score each candidate
select or soft-aggregate
```

First scorer target:

```text
negative action MSE through the frozen decoder
```

because the current bottleneck is action value, not geometric reconstruction
alone.

Required comparisons:

- prior mean;
- random sample mean;
- oracle best-of-K action;
- learned scorer argmax;
- learned scorer soft aggregation.

Promotion condition:

```text
learned scorer beats prior mean and random sample mean
with reduced regret to oracle best-of-K
in both seeds and without gripper collapse.
```

### Gate 2.5: Contact / Gripper / Phase Diagnostics

Add lightweight non-object-`SE(3)` task structure:

- gripper transition labels;
- contact-proxy windows;
- phase bins from `u_geom`;
- temporal rank pairs;
- optional object mask or depth signal.

This is the GuidedVLA lesson adapted to GeoMoCo-WM:

```text
guide the decoder with factors,
do not hope action MSE alone discovers them.
```

### Gate 2.6: Guided Decoder Branch

Test a small ControlNet-style or attention-guidance branch:

```text
base action decoder / policy path
+ zero-initialized GeoMoCo guidance branch
```

Candidate guided factors:

- future-motion sample;
- progress / phase embedding;
- composition confidence;
- scorer value;
- gripper/contact readout.

This branch should be evaluated against simple concatenation. The point is not
capacity; the point is factor attribution.

### Gate 3: Strong Action Head Baselines

Only after Gate 2.4d / 2.5 / 2.6:

- Diffusion Policy;
- Mean Flow Policy / MVP-style one-step head;
- SDP-style `SE(3)` geometry-aware policy branch;
- possibly GuidedVLA-style action attention specialization if a large VLA
  backbone is introduced.

## Baseline Map

Use the papers as baselines by role, not as a single flat list.

| Role | Baseline / Reference |
| --- | --- |
| direct policy | direct context MLP / transformer / `pi0`-style branch |
| future EEF trajectory interface | OASIS-style oracle and learned EEF trajectory |
| visual motion latent | ZipMo-style visual motion embedding |
| actionless prior + inverse dynamics | AMPLIFY-style decomposition |
| train-time WAM action branch | Fast-WAM-style direct action head with world-model training |
| policy-side geometry | SDP / 3D Diffuser Actor / 3D FlowMatch Actor |
| fast action generator | Mean Flow Policy / MVP |
| factor-guided action decoder | GuidedVLA-style object/depth/skill heads |
| candidate readout | BCQ / IBC / Trajectory Transformer / Diffuser / Visual MPC family |

## Paper Framing

A clean future paper story would be:

```text
Existing VLA/WAM policies either:
  generate actions directly,
  predict visual futures,
  predict EEF trajectories,
  or specialize action decoders with task factors.

GeoMoCo-WM instead asks:
  can we learn a visually grounded, stochastic, compositional motion state
  whose sampled future rollouts can be read out into action-useful candidates?
```

The key novelty should be:

- structured future-motion state;
- stochastic sample coverage;
- deployable sample readout;
- phase/progress/composition diagnostics;
- controlled motion-to-action attribution.

Do not overclaim:

- not full object-centric world modeling yet;
- not robot-object relative `SE(3)` at scale yet;
- not a replacement for VLA policies;
- not closed-loop superiority until rollouts verify it.

## Stop Rules

Stop or downgrade claims if:

- learned sample readout cannot beat prior mean;
- gains only exist under oracle best-of-K;
- gripper/contact metrics regress;
- direct context or DINO-only baselines erase the motion prior advantage;
- a stronger action head explains the gains without GeoMoCo factors;
- closed-loop results do not preserve offline action gains.

## Working Summary

The recent papers converge on one design principle:

```text
motion, geometry, phase, object grounding, and action decoding should be
separated enough to test, but connected enough to execute.
```

GeoMoCo-WM should keep that separation:

```text
visual grounding -> stochastic future-motion prior -> sample readout
-> controlled action decoder -> later strong policy head
```

This keeps the project distinct from OASIS-style EEF trajectory policies,
GuidedVLA-style VLA finetuning, Fast-WAM-style action denoising, and
ZipMo / AMPLIFY-style visual motion priors, while still borrowing their best
interfaces.
