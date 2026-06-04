# GeoMoCo-cVAE Experiment Plan

## 2026-06-04 Route Update

This plan has been refined by
`docs/ideas_plans/plans/visual-grounded-geomoco-wm-plan.md`.

The current first-pass route is no longer a state-only cVAE plus optional
ZipMo/AMPLIFY comparison. The project should first build:

```text
DINO visual grounding
  -> GeoMoCo-cVAE future geometric motion prior
  -> controlled action decoder
```

ZipMotion and AMPLIFY are deferred to follow-up modules or optional baselines.
They should not be first-pass main-method dependencies unless the paper claims
superiority over visual motion-token or actionless-video methods.

## Decision

Start a clean `Geomoco-WM` project for the world-motion line, while preserving
the existing GeoMoCo work as a baseline/reference implementation.

The new project should focus on a visually grounded structured stochastic
future-motion prior rather than a broad new action-policy stack.

## Core Story

1. Encode RGB history with DINO and task/EEF/proprio-conditioned pooling.
2. Fuse visual grounding, history, proprioception, and task context.
3. Model a conditional distribution over future geometry-aware motion latents.
4. Decode sampled latents into future SE(3), phase/progress, and optional visual
   feature targets.
5. Convert the motion representation into executable action chunks with a
   controlled inverse-dynamics decoder.
6. Evaluate predictive and control value over state-only, DINO-only,
   deterministic GeoMoCo, random/shuffled latent, and oracle future-motion
   controls.

## Required Comparisons

- Direct DINO/proprio/task BC or ACT-style action policy.
- Direct DINO/proprio/task Diffusion Policy or MeanFlow-style action decoder if
  budget allows.
- GeoMoCo-AE with deterministic latent.
- GeoMoCo-cVAE with sampled latent.
- GeoMoCo-cVAE without composition loss.
- GeoMoCo-cVAE without stochastic sampling.
- Direct residual future-latent predictor.
- Random and shuffled latent controls.
- Oracle future motion into the same action decoder.
- ZipMo / AMPLIFY adapters only as deferred baselines or follow-up modules.

The controlled comparison is the most important result: keep the action decoder,
data, views, horizons, seeds, and evaluation suite fixed while changing only the
motion representation.

## Minimal Closed Loop

```text
LIBERO demonstration
  -> MotionChunkDataset
  -> DINO visual grounding
  -> GeoMoCo-AE / GeoMoCo-cVAE future-motion prior
  -> shared action decoder
  -> offline metrics + small closed-loop rollout
```

## Fairness Constraints

- Same LIBERO suite.
- Same image views and proprioceptive inputs.
- Same number of demonstrations.
- Same action horizon and rollout seeds.
- No privileged object state unless all methods receive it.
- Comparable decoder capacity for controlled baselines.

## Success Criteria

- Reconstruction and rollout metrics both improve, not only reconstruction.
- Gains are strongest in low-data, multimodal, and long-horizon phase tasks.
- Oracle future-motion decoder provides a meaningful upper bound.
- Ablations show the value of stochastic sampling and geometric composition.
