# GeoMoCo-cVAE Experiment Plan

## Decision

Start a clean `Geomoco-WM` project for the world-motion line, while preserving
the existing GeoMoCo work as a baseline/reference implementation.

The new project should focus on a structured stochastic motion prior rather than
a broad new action-policy stack.

## Core Story

1. Encode observation/history/task context.
2. Model a conditional distribution over geometry-aware motion latents.
3. Decode sampled latents into future SE(3) motion chunks.
4. Convert the motion representation into executable action chunks with a
   lightweight inverse-dynamics decoder.
5. Evaluate whether this improves closed-loop manipulation over direct BC and
   non-geometric motion-token baselines.

## Required Comparisons

- Direct BC / ACT-style or diffusion-style action policy.
- GeoMoCo-AE with deterministic latent.
- GeoMoCo-cVAE with sampled latent.
- GeoMoCo-cVAE without composition loss.
- GeoMoCo-cVAE without stochastic sampling.
- AMPLIFY official policy if reproducible.
- AMPLIFY tokens with the same action decoder.
- ZipMo official policy if reproducible.
- ZipMo latent with the same action decoder.
- Oracle future motion into the same action decoder.

The controlled comparison is the most important result: keep the action decoder,
data, views, horizons, seeds, and evaluation suite fixed while changing only the
motion representation.

## Minimal Closed Loop

```text
LIBERO demonstration
  -> MotionChunkDataset
  -> GeoMoCo-AE / GeoMoCo-cVAE
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
