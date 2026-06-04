# Visual-Grounded GeoMoCo-WM Plan

## Decision

The first serious `Geomoco-WM` track should move directly to visual grounding,
but it should use a lightweight DINO-based front-end before attempting full
ZipMotion or AMPLIFY integration.

Current positioning:

```text
DINO visual grounding
  -> GeoMoCo-cVAE future geometric motion prior
  -> controlled action decoder / inverse dynamics
```

ZipMotion and AMPLIFY should remain follow-up modules or optional baselines, not
first-pass dependencies. This keeps the project attributable while still making
the method more than a VAE version of old GeoMoCo.

## Why The Route Changed

Old GeoMoCo evidence from `/home/user/projects/se3-group-motion-control` showed:

- pure GeoMoCo latent was not enough for policy or retrieval success;
- frozen GeoMoCo latent could improve offline action prediction but fail or hurt
  closed-loop control;
- EEF-centric `SE(3)` plus pooled latent was better interpreted as a
  phase/composition factor than as a full world state;
- the strongest positive practical signal was predictive state-factor value,
  not standalone policy success.

Therefore, `Geomoco-WM` cannot just be:

```text
old GeoMoCo latent + cVAE sampling + action head
```

It must add deployable grounding. Visual grounding is the cleanest way to show
why GeoMoCo is valuable: the method should convert visual evidence into a
structured, compositional, executable future-motion state.

## First-Principles Thesis

Manipulation requires three different objects:

```text
G_t:
  grounded current state from vision/proprio/task

z_geo:
  compositional geometric motion factor

a_t:
  commanded intervention / robot action
```

Old GeoMoCo mainly studied:

```text
observed motion segment -> z_geo
z_A, z_B -> z_AB
```

GeoMoCo-WM should study:

```text
visual/proprio/task context
  -> p(z_future | context)
  -> future geometric motion target
  -> state prediction / action decoding / closed-loop validation
```

The key distinction is prospective prediction. The model must predict or sample
future motion states before they are observed.

## Architecture

### 1. Visual Grounding Front-End

Use DINO as the first visual backbone.

```text
RGB history from agentview/wrist
  -> frozen DINO patch tokens
  -> temporal attention or token-difference module
  -> EEF/proprio/task-conditioned pooling
  -> visual grounding token g_t
```

DINO is not a direct replacement for ZipMotion. It supplies strong visual
features and correspondences, while the project must add the temporal/motion
pooling layer needed for grounding.

Initial recommendation:

- freeze DINO first;
- cache features for reproducibility;
- use agentview and wrist when available;
- condition pooling on EEF pose, gripper, proprioception, and task embedding;
- add optional future visual-feature prediction to prevent gripper-only
  shortcuts.

### 2. GeoMoCo-cVAE Future-Motion Prior

The cVAE should predict future motion, not merely regularize old latents.

Candidate targets:

```text
z_B:
  next motion segment latent

z_AB:
  longer-horizon composed future latent

T_future:
  future EEF SE(3) delta / endpoint transform

u_future:
  normalized geometric motion progress or phase

phi_future:
  frozen visual feature target, optional
```

Core model:

```text
context c_t = [g_t, proprio_t, EEF_t, gripper_t, task, history]

q(z | c_t, future_motion)
p(z | c_t)
decoder(c_t, z) -> future geometric motion target
```

Important: `p(z | c_t)` is the deployable prior. The posterior is only a
training-time path.

### 3. Motion Decoding And Composition

Keep the GeoMoCo identity visible:

```text
z_A, z_B_pred -> compose -> z_AB_pred
```

Also keep direct future-latent prediction as a hard baseline:

```text
context -> z_AB_direct
```

The claim should not be that explicit composition always has the lowest MSE.
The claim should be that it provides an auditable, calibratable future-motion
prior with useful predictive/control value.

### 4. Action Decoder / Inverse Dynamics

AMPLIFY is not required for the first version. The action bridge can be:

```text
context + future motion target -> action chunk
```

Recommended order:

1. MLP/Transformer action-chunk head for fast attribution.
2. Diffusion-Policy-style action head as a strong fixed decoder.
3. MeanFlow / flow-style action head as a faster generative decoder if useful.

The decoder must be shared across representation baselines. Otherwise any gain
could come from the action model rather than GeoMoCo.

## First-Pass Dataset Contract

Each training item should expose:

```text
rgb_history:
  agentview and/or wrist frames

proprio:
  robot joints if available, EEF pose, gripper

task:
  task id or language embedding

history_motion:
  recent EEF SE(3), gripper, action history

future_motion:
  future EEF SE(3), future geometric progress, future GeoMoCo target

action_chunk:
  future robot actions

object_state_teacher:
  optional privileged field for diagnostics, teacher heads, and upper bounds
```

Main deployable baselines should not require object state at inference. Object
state can still be used as a teacher, diagnostic, or upper-bound control.

## Experimental Gates

### Gate 0: Data And Feature Sanity

- Export LIBERO demonstrations into the new visual-grounded dataset contract.
- Cache DINO features.
- Verify action, EEF, gripper, task, and RGB alignment.
- Add nonblank image and feature-shape audits.

### Gate 1: Visual Grounding Probe

Compare:

- proprio/task/history only;
- DINO only;
- DINO + proprio/task/history;
- DINO + shuffled temporal order;
- DINO + EEF/task-conditioned pooling.

Targets:

- future EEF delta;
- geometric progress;
- optional future DINO feature.

### Gate 2: GeoMoCo-AE / cVAE Future-Motion Prior

Compare:

- deterministic GeoMoCo-AE;
- GeoMoCo-cVAE sampled prior;
- cVAE without composition loss;
- cVAE without stochastic sampling;
- direct residual future-latent predictor;
- random latent control;
- shuffled latent control;
- oracle future motion.

Metrics:

- future motion reconstruction;
- KL/calibration;
- diversity vs mode collapse;
- future state/progress prediction;
- composition closure;
- hard-negative phase separation.

### Gate 3: Shared Action Decoder

Use the same decoder capacity for all inputs:

- DINO + proprio + task direct policy;
- DINO + proprio + task + GeoMoCo-AE;
- DINO + proprio + task + GeoMoCo-cVAE;
- DINO + proprio + task + random/shuffled latent;
- DINO + proprio + task + oracle future motion.

Action heads:

- start with action chunk MLP/Transformer;
- then optionally repeat the same controlled comparison with Diffusion Policy
  or MeanFlow-style decoder.

### Gate 4: Closed-Loop LIBERO

Only run closed-loop promotion after Gates 1-3 show non-degenerate value.

Use fixed:

- suite;
- views;
- action horizon;
- seeds/init states;
- decoder capacity;
- training demos;
- no privileged object state unless explicitly testing an upper bound.

Closed-loop should initially be a small validation slice, not the only evidence.

## Baseline Policy

Required first-pass baselines:

- direct visual BC / action-chunk policy;
- visual Diffusion Policy or MeanFlow-style decoder if budget allows;
- DINO-only future predictor;
- GeoMoCo-AE;
- GeoMoCo-cVAE;
- random/shuffled latent controls;
- direct future-latent residual baseline;
- oracle future motion upper bound.

Deferred baselines:

- ZipMotion visual motion front-end;
- AMPLIFY-style actionless video prior;
- Fast-WAM-style video co-training.

These deferred baselines become necessary only if the paper claims superiority
over visual motion-token methods or actionless video-motion priors.

## Paper Boundary

The first paper should not claim:

```text
full visual world model;
general VLA policy;
ZipMotion/AMPLIFY replacement;
closed-loop SOTA;
object-centric grounding without object labels.
```

The first paper can claim:

```text
visual-grounded conditional geometric future-motion prior;
DINO features can be compressed into a structured GeoMoCo state;
GeoMoCo-cVAE improves predictive state/progress modeling over deterministic,
state-only, DINO-only, random, and shuffled controls;
the prior can be connected to action decoders under a fixed interface.
```

## Immediate Implementation Slices

1. Add visual-grounded dataset schema and exporter.
2. Add DINO feature cache and visual grounding module.
3. Train Gate-1 visual grounding probes.
4. Extend GeoMoCo-cVAE context to include visual grounding token `g_t`.
5. Add direct residual future-latent baseline.
6. Add shared action-chunk decoder comparison.
7. Promote to diffusion/flow action head only after the simple decoder gives a
   meaningful attribution signal.
