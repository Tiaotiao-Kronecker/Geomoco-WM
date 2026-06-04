# Active Worklog

- Last updated: 2026-06-04
- Repository: `/home/user/projects/Geomoco-WM`

## Current Objective
- Build `Geomoco-WM` as a clean project for the GeoMoCo world-motion line:
  structured GeoMoCo-AE / GeoMoCo-cVAE motion priors, a lightweight action
  decoder, and controlled ZipMo / AMPLIFY representation comparisons.

## Current Execution Slice
- Initial repository scaffolding and traceability setup.
- Keep the first implementation small: data interfaces, model skeletons,
  baseline adapter boundaries, minimal LIBERO config, and experiment plan.

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

## Current Interpretation
- The repository is ready as a clean method track and handoff point.
- The current code proves only package structure and minimal tensor plumbing.
  It does not yet prove dataset export, training stability, reconstruction
  quality, closed-loop performance, or baseline fairness.

## Open Decisions Or Blockers
- Decide the first LIBERO suite and demo budget for the minimal closed loop.
- Define the exact `MotionChunkDataset` export contract from demonstrations.
- Choose context encoder inputs: image views, proprioception, task text, and
  history length.
- Decide whether ZipMo / AMPLIFY are compared through official checkpoints,
  same-decoder controlled baselines, or both in the first pass.
- Install project dev dependencies if `pytest` should be part of local checks.

## Next Session Entry Point
1. Implement the LIBERO demonstration exporter into `MotionChunkDataset`.
2. Add the first training script for GeoMoCo-AE and GeoMoCo-cVAE reconstruction.
3. Add shared action-decoder training with BC, oracle motion, AE, and cVAE
   inputs under the same horizon and capacity.
