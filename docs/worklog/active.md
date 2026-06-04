# Active Worklog

- Last updated: 2026-06-04
- Repository: `/home/user/projects/Geomoco-WM`

## Current Objective
- Build `Geomoco-WM` as a visual-grounded GeoMoCo world-motion project:
  DINO visual grounding, GeoMoCo-AE / GeoMoCo-cVAE future-motion priors, and a
  controlled action decoder for predictive and closed-loop validation.

## Current Execution Slice
- Plan refinement and archive after deciding that the first serious route should
  use visual grounding rather than a state-only VAE version of old GeoMoCo.
- Keep ZipMotion and AMPLIFY as deferred modules or optional baselines, while
  using DINO as the first lightweight visual grounding front-end.

## Latest Results
- Local project created at `/home/user/projects/Geomoco-WM`.
- GitHub remote configured as `git@github.com:Tiaotiao-Kronecker/Geomoco-WM.git`.
- Initial scaffold pushed to `origin/main` at commit `3718f19`.
- Core artifacts:
  - `README.md`
  - `pyproject.toml`
  - `src/geomoco_wm/models/geomoco_ae.py`
  - `src/geomoco_wm/models/geomoco_cvae.py`
  - `src/geomoco_wm/models/action_decoder.py`
  - `src/geomoco_wm/integrations/zipmo_adapter.py`
  - `src/geomoco_wm/integrations/amplify_adapter.py`
  - `experiments/geomoco_cvae/configs/minimal_libero.yaml`
  - `docs/ideas_plans/plans/geomoco-cvae-experiment-plan.md`
- Verification completed:
  - `python -m compileall src tests`
  - PyTorch smoke test for `GeoMoCoCVAE` and `ActionDecoder`
- `pytest` was not run because the active environment does not have `pytest`
  installed.
- 2026-06-04 route refinement:
  - old GeoMoCo evidence suggests EEF-centric latent is a phase/composition
    factor, not a full policy or world state;
  - a state-only cVAE would risk becoming only a multimodal version of old
    GeoMoCo;
  - current plan moves to DINO visual grounding plus GeoMoCo-cVAE future-motion
    prior;
  - Diffusion Policy / MeanFlow-style action heads are allowed as stronger
    shared decoders, but AMPLIFY is not required for the first version;
  - ZipMotion is deferred because DINO is a lighter first grounding front-end,
    while ZipMotion remains a stronger visual-motion extension/baseline.
- Added plan artifacts:
  - `docs/ideas_plans/plans/visual-grounded-geomoco-wm-plan.md`
  - `experiments/geomoco_cvae/configs/visual_grounded_libero.yaml`

## Current Interpretation
- The repository is ready as a clean visual-grounded method track and handoff
  point.
- The current code proves only package structure and minimal tensor plumbing.
  It does not yet prove visual feature export, grounding quality, cVAE training
  stability, predictive value, closed-loop performance, or baseline fairness.
- The project should not claim full world-model status until it predicts or
  samples future motion from visual/proprio/task context and shows value in
  predictive or controlled action-decoder gates.

## Open Decisions Or Blockers
- Decide the first LIBERO suite and demo budget for visual-grounded gates.
- Define the exact visual-grounded dataset contract, including RGB history,
  DINO feature cache, proprioception, EEF pose, gripper, task, future motion,
  action chunk, and optional object-state teacher fields.
- Choose DINO backbone/version and feature-cache format.
- Decide the first action decoder: simple action-chunk transformer first, with
  Diffusion Policy / MeanFlow-style decoder only after attribution is clear.
- Decide whether object state is teacher/diagnostic only or included in any
  upper-bound baseline.
- Install project dev dependencies if `pytest` should be part of local checks.

## Next Session Entry Point
1. Implement the visual-grounded LIBERO exporter and DINO feature-cache path.
2. Add Gate-1 visual grounding probes for future EEF SE(3), geometric progress,
   and optional future DINO feature prediction.
3. Extend GeoMoCo-cVAE context with visual grounding token `g_t` and add
   deterministic AE, stochastic cVAE, direct residual, random, shuffled, and
   oracle future-motion baselines.
