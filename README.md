# Geomoco-WM

Geometry-aware world-motion modeling for manipulation.

This repository starts the clean GeoMoCo world-model line discussed in the
planning notes: a structured conditional motion prior, paired with a lightweight
action decoder, for closed-loop manipulation evaluation.

## Research Thesis

GeoMoCo should be evaluated as a stochastic world-motion state, not only as an
offline SE(3) motion embedding. The core claim is:

> A geometry-aware conditional motion VAE can serve as a compact stochastic
> world-motion prior. When paired with a lightweight inverse-dynamics action
> decoder, it improves manipulation success and sample efficiency over direct
> behavior cloning, deterministic motion embeddings, and visual motion-token
> baselines.

## Initial Scope

- GeoMoCo-AE baseline: deterministic geometric motion embedding.
- GeoMoCo-cVAE: conditional distribution over geometric motion latents.
- Motion chunk decoder: latent to future SE(3) motion chunk.
- Lightweight action decoder: observation, proprioception, task, and motion
  latent or decoded motion chunk to action horizon.
- Controlled representation comparisons against ZipMo and AMPLIFY adapters
  using the same action decoder.
- Small closed-loop LIBERO rollout path before scaling.

This project intentionally avoids becoming a new large VLA policy. The policy
component exists to verify whether the world-motion latent is executable and
useful for control.

## Repository Layout

```text
src/geomoco_wm/
  data/
    motion_chunk_dataset.py
  models/
    geomoco_ae.py
    geomoco_cvae.py
    action_decoder.py
  integrations/
    zipmo_adapter.py
    amplify_adapter.py

experiments/geomoco_cvae/configs/
  minimal_libero.yaml

docs/ideas_plans/plans/
  geomoco-cvae-experiment-plan.md
```

## First Milestones

1. Export LIBERO demonstration trajectories into `MotionChunkDataset`.
2. Train GeoMoCo-AE and GeoMoCo-cVAE on future SE(3) motion chunks.
3. Train one shared lightweight action decoder for all motion representations.
4. Run controlled baselines: BC, GeoMoCo-AE, GeoMoCo-cVAE, ZipMo, AMPLIFY,
   oracle future motion.
5. Add closed-loop LIBERO evaluation and ablations for stochastic sampling,
   composition loss, and action horizon.

## Development

```bash
pip install -e ".[dev]"
pytest
```
