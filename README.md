# Geomoco-WM

Geometry-aware world-motion modeling for manipulation.

This repository starts the clean GeoMoCo world-motion line discussed in the
planning notes: a visually grounded conditional geometric motion prior, paired
with a controlled action decoder, for manipulation world-model interfaces.

## Research Thesis

GeoMoCo should be evaluated as a visually grounded stochastic future-motion
prior, not only as an offline SE(3) motion embedding. The core claim is:

> A DINO-grounded GeoMoCo-cVAE can convert visual context, proprioception, task,
> and motion history into a compact distribution over future geometric motion
> states. This distribution should improve future state/progress prediction and
> serve as an executable motion target for controlled action decoders.

## Initial Scope

- DINO visual grounding front-end: frozen visual features plus
  EEF/proprio/task-conditioned temporal pooling.
- GeoMoCo-AE baseline: deterministic grounded geometric motion embedding.
- GeoMoCo-cVAE: conditional distribution over future geometric motion latents
  or future `B` / `AB` motion proposals.
- Motion decoder: latent to future EEF SE(3), geometric progress, and optional
  visual feature targets.
- Controlled action decoder: MLP/transformer chunk head first, with
  Diffusion-Policy or MeanFlow-style heads as stronger fixed decoders later.
- ZipMotion and AMPLIFY remain follow-up or optional baselines, not first-pass
  main-method dependencies.
- Small closed-loop LIBERO rollout path after predictive and offline gates pass.

This project intentionally avoids becoming a new large VLA policy. The action
component exists to verify whether the visual-grounded world-motion prior is
executable and useful for control.

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

1. Export LIBERO demonstrations with RGB history, proprioception, EEF pose,
   gripper, task, action chunks, and optional object-state teacher fields.
2. Cache frozen DINO visual tokens and train the first visual grounding module.
3. Train GeoMoCo-AE and GeoMoCo-cVAE on future geometric motion proposals.
4. Run predictive gates: future state/progress/contact/visual-feature probes
   against state-only, DINO-only, random-latent, shuffled-latent, AE, and cVAE
   baselines.
5. Train one shared action decoder for all motion representations before trying
   stronger Diffusion-Policy or MeanFlow-style action heads.
6. Add closed-loop LIBERO evaluation only after predictive and offline action
   gates show non-degenerate value.

## Development

```bash
pip install -e ".[dev]"
pytest
```
