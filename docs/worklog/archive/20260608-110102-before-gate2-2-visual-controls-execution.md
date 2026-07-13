# Active Worklog

- Last updated: 2026-06-08
- Repository: `/home/user/projects/Geomoco-WM`

## Current Objective
- Build `Geomoco-WM` as a visual-grounded GeoMoCo world-motion project:
  DINO visual grounding, GeoMoCo-AE / GeoMoCo-cVAE future-motion priors, and a
  controlled action decoder for predictive and closed-loop validation.

## Current Execution Slice
- Gate 1.6 two-file four-suite oracle action-decoder geodesic replacement
  completed.
- Gate 2 deterministic context-only future-motion prior completed as a
  diagnostic baseline.
- Gate 2.1 suite/task-conditioned future-motion prior completed as a stronger
  diagnostic baseline; it improves over Gate 2 but still does not beat direct
  context as a policy interface.
- Gate 2.2a DINOv2 global visual future-motion prior completed as the first
  learned prior that beats direct context through the frozen action-decoder
  interface.
- Gate 2.2b patch-pooled DINOv2 cross-attention future-motion prior completed
  and improves over Gate 2.2a.
- Action Semantics Audit completed and the action metric contract is now
  upgraded from normalized SE(3)-aware split metrics to physical translation
  plus SO(3) geodesic rotation metrics.
- Next run target: run shuffled-vision and camera-ablation controls before
  moving to visual-grounded cVAE.
- The exporter writes HDF5 episode references plus lightweight future
  EEF/action windows; DINO global features are now cached separately for
  Gate 2.2a.

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
- 2026-06-04 design-gate discussion archived:
  - `docs/agent_qa/2026-06-04-geomoco-wm-design-gates.md`
  - core decision: do not promote GeoMoCo-cVAE or the action-decoder route until
    predictive gates, future-motion coverage gates, and the oracle
    future-motion action-decoder gate show non-degenerate value.
- 2026-06-05 complete experiment blueprint archived:
  - `docs/ideas_plans/plans/geomoco-wm-complete-experiment-blueprint.md`
  - `docs/ideas_plans/html/geomoco-wm-complete-experiment-blueprint.html`
  - content covers experiment order, network architecture, dataset/window
    contract, DINO failure lessons, LIBERO-Long/LIBERO-10 promotion criteria,
    baseline matrix, metrics, and stop rules.
- 2026-06-05 Gate 0 HDF5 inspection slice written for review:
  - `src/geomoco_wm/data/libero_hdf5_inspect.py`
  - `scripts/inspect_libero_hdf5.py`
  - `tests/test_libero_hdf5_inspect.py`
  - `docs/ideas_plans/plans/gate0-libero-hdf5-inspection-runbook.md`
- Gate 0 verification completed on synthetic HDF5 only:
  - `python -m compileall src scripts tests`
  - `python -m unittest discover -s tests -p 'test_libero_hdf5_inspect.py'`
  - result: 4 tests passed.
- Gate 0 smoke inspection executed on real local LIBERO data:
  - command: `python scripts/inspect_libero_hdf5.py --input-path /home/user/dataset/libero_official/libero_goal --suite-name libero_goal --max-files 1 --max-demos-per-file 2 --output-json outputs/gate0/libero_goal_hdf5_inspection_smoke.json --output-md outputs/gate0/libero_goal_hdf5_inspection_smoke.md`
  - result: 1 file, 2 demos, 276 frames;
  - readiness: visual grounding, dual camera, EEF motion, action chunks, and
    proprio context all supported;
  - warning: object-state teacher fields unavailable, keep diagnostic-only.
- Gate 0 full `libero_goal` metadata inspection executed:
  - report: `outputs/gate0/libero_goal_hdf5_inspection.md`
  - JSON: `outputs/gate0/libero_goal_hdf5_inspection.json`
  - result: 10 files, 500 demos, 63,728 frames;
  - demo length range: 75 to 347, mean 127.456;
  - all required fields present, all sequence lengths aligned, all actions 7D,
    all EEF states 6D, all gripper states 2D, all joint states 7D;
  - `supports_gate0_dataset_export: true`;
  - `supports_object_state_teacher: false`.
- 2026-06-05 Gate 1 first exporter implemented:
  - `src/geomoco_wm/data/libero_hdf5_export.py`
  - `scripts/export_libero_windows.py`
  - `tests/test_libero_hdf5_export.py`
  - `docs/ideas_plans/plans/gate1-libero-window-export-runbook.md`
- Gate 1 verification completed:
  - `python -m compileall src scripts tests`
  - `python -m unittest discover -s tests -p 'test_libero_hdf5_*.py'`
  - result: 7 tests passed.
- Tiny real-data exporter smoke completed:
  - command: `python scripts/export_libero_windows.py --input-path /home/user/dataset/libero_official/libero_goal --suite-name libero_goal --output-dir outputs/libero_windows/libero_goal_smoke --context-len 2 --horizon 8 --stride 4 --max-files 1 --max-demos-per-file 1 --max-windows 3`
  - output: `outputs/libero_windows/libero_goal_smoke/`
  - result: 1 episode record, 3 window records, 138 source frames;
  - expected warning: `max_windows` reached.
- Experiment-facing oracle action-decoder path added:
  - `src/geomoco_wm/data/window_dataset.py`
  - `scripts/train_oracle_action_decoder.py`
  - dry-run result on smoke windows: 3 windows, context dim 15, motion dim 48,
    action dim 7, horizon 8;
  - 1-epoch CPU smoke completed and wrote
    `outputs/oracle_action_decoder/libero_goal_smoke/metrics.json`;
  - this is only a loop smoke, not a meaningful performance result.
- Motion-to-action and decoder discussion archived:
  - `docs/agent_qa/2026-06-05-motion-to-action-and-decoder-plan.md`
  - decision: use MLP as the first attribution-clean diagnostic, add stronger
    temporal/diffusion/flow decoders only after the oracle future-motion gate
    has signal.
- 10-demo drawer export and oracle-vs-direct MLP smoke completed:
  - export: `outputs/libero_windows/libero_goal_task0_10demo/`
  - result: 10 demos, 307 windows, 1,383 frames;
  - direct-context MLP final val MSE / MAE: `0.035896` / `0.113958`;
  - GT future EEF delta MLP final val MSE / MAE: `0.017163` / `0.088143`;
  - relative validation improvement: 52.19% MSE reduction, 22.65% MAE
    reduction;
  - report: `docs/agent_qa/2026-06-05-oracle-action-decoder-10demo-smoke.md`.
- 2026-06-06 local LIBERO data expanded from `libero_goal` to the standard
  four-suite set under `/home/user/dataset/libero_official`:
  - `libero_spatial`: 10 HDF5 files, 5.9G, `hdf5_ok=10`;
  - `libero_object`: 10 HDF5 files, 7.0G, `hdf5_ok=10`;
  - `libero_goal`: 10 HDF5 files, 6.0G, `hdf5_ok=10`;
  - `libero_10`: 10 HDF5 files, 13G, `hdf5_ok=10`.
- 2026-06-06 four-suite batch exporter and oracle input support added:
  - `src/geomoco_wm/data/libero_hdf5_export.py` now has
    `export_libero_hdf5_suite_collection`;
  - `scripts/export_libero_windows.py` now supports `--all-libero-suites`,
    `--suite-names`, per-suite caps, per-suite outputs, and combined
    `episodes.jsonl` / `windows.jsonl`;
  - `src/geomoco_wm/data/window_dataset.py` and
    `scripts/train_oracle_action_decoder.py` now support one or more
    `windows.jsonl` inputs and record suite/task counts;
  - oracle train/val split can now use `--split-by episode` for real
    comparisons.
- Four-suite smoke completed:
  - export:
    `outputs/libero_windows/libero_all_suites_smoke/`;
  - result: 4 suites, 4 files, 4 episodes, 8 windows, 656 frames;
  - dry-run future-delta spec: 8 windows, context dim 15, motion dim 48,
    action dim 7, horizon 8, suite counts 2 each;
  - dry-run direct-context spec: same dataset, motion dim 0;
  - 1-epoch loop smokes wrote:
    `outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta_1epoch/`
    and
    `outputs/oracle_action_decoder/libero_all_suites_smoke_direct_context_1epoch/`.
- Episode-level split and four-suite slice discussion archived:
  - `docs/agent_qa/2026-06-06-episode-split-and-four-suite-slice.md`;
  - decision: promote direct-context vs GT-future-motion from single-task smoke
    to four-suite episode-level comparison before cVAE/DINO claims.
- Four-suite test dataset usage archived:
  - `docs/agent_qa/2026-06-06-four-suite-test-dataset-usage.md`;
  - local dataset root: `/home/user/dataset/libero_official`;
  - standard suites present and validated: `libero_spatial`, `libero_object`,
    `libero_goal`, and `libero_10`, each with 10 readable HDF5 files;
  - dataset-use policy: smoke checks plumbing only, small formal slice uses
    one task file per suite with all demos, and full four-suite export waits
    until JSONL size and training speed are acceptable.
- Four-suite small formal slice exported:
  - output: `outputs/libero_windows/libero_all_suites_1file_all_demos_h8/`;
  - result: 4 suites, 4 HDF5 task files, 200 demos, 7,921 windows, 33,201
    frames;
  - no dropped short episodes and no exporter warnings;
  - combined `windows.jsonl` size: 33M.
- Four-suite oracle action-decoder comparison completed:
  - report:
    `docs/agent_qa/2026-06-06-four-suite-oracle-action-results.md`;
  - split: episode-level;
  - model: MLP `ActionDecoder`, hidden dims `256,256`, 20 epochs,
    batch size 64;
  - seed 7 direct-context val MSE / MAE: `0.081479` / `0.160300`;
  - seed 7 GT-future-motion val MSE / MAE: `0.035064` / `0.085608`;
  - seed 7 relative reduction: 56.97% MSE, 46.60% MAE;
  - seed 17 direct-context val MSE / MAE: `0.065460` / `0.149165`;
  - seed 17 GT-future-motion val MSE / MAE: `0.031770` / `0.084627`;
  - seed 17 relative reduction: 51.47% MSE, 43.27% MAE.
- Canonical experiment log directory added:
  - `docs/experiments/README.md`;
  - first formal run record:
    `docs/experiments/runs/2026-06-06_gate1_5_four_suite_oracle_action_decoder.md`;
  - reusable template:
    `docs/experiments/templates/run-record-template.md`;
  - policy: `docs/agent_qa/` keeps discussion context, while
    `docs/experiments/` is the cleaned ledger for run configs, metrics,
    artifacts, interpretation, limits, and next decisions.
- SE(3)-aware action metrics added to the oracle action decoder script:
  - code: `scripts/train_oracle_action_decoder.py`;
  - test: `tests/test_oracle_action_decoder_metrics.py`;
  - metrics now include translation, rotation, combined first-six-dim
    `se3`, and gripper MSE/MAE/L2-style decompositions in addition to flat
    MSE/MAE.
- Verification after metric changes:
  - `python -m compileall scripts tests src`;
  - `python -m unittest discover -s tests -p 'test_oracle_action_decoder_metrics.py'`;
  - `python -m unittest discover -s tests -p 'test_libero_hdf5_*.py'`;
  - `PYTHONPATH=/home/user/projects/Geomoco-WM/src python -m unittest discover -s tests`;
  - result: all relevant tests passed; full discover passes 10 tests when
    `src` is on `PYTHONPATH`.
- Gate 1.6 two-file four-suite slice exported:
  - output: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/`;
  - result: 4 suites, 8 HDF5 task files, 400 demos, 8 tasks, 16,518 windows,
    69,073 frames;
  - no dropped short episodes and no exporter warnings;
  - combined `windows.jsonl` size: 68M.
- Gate 1.6 oracle action-decoder comparison completed on CUDA:
  - formal run record:
    `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_se3_metrics.md`;
  - scale-up comparison:
    `docs/experiments/comparisons/2026-06-06_oracle_action_gate_scaleup.md`;
  - split: episode-level;
  - model: MLP `ActionDecoder`, hidden dims `256,256`, 20 epochs,
    batch size 64, `--device cuda`;
  - seed 7 direct-context val MSE / MAE: `0.068109` / `0.148911`;
  - seed 7 GT-future-motion val MSE / MAE: `0.033542` / `0.082386`;
  - seed 7 relative reduction: 50.75% MSE, 44.67% MAE;
  - seed 17 direct-context val MSE / MAE: `0.063910` / `0.145336`;
  - seed 17 GT-future-motion val MSE / MAE: `0.029407` / `0.076629`;
  - seed 17 relative reduction: 53.99% MSE, 47.27% MAE;
  - SE(3) MSE reduction: 82.34% for seed 7, 83.58% for seed 17;
  - gripper MSE reduction: 28.19% for seed 7, 25.26% for seed 17.
- Default restricted execution context did not expose CUDA:
  - `torch 2.10.0+cu128`;
  - `torch.cuda.is_available() == False`;
  - `torch.cuda.device_count() == 0`;
  - this was acceptable for the small CPU MLP gate.
- Elevated GPU checks confirmed the machine and Python environment can see the
  5090:
  - `nvidia-smi`: `NVIDIA GeForce RTX 5090`, driver `580.95.05`, system CUDA
    `13.0`;
  - elevated Python: `torch.cuda.is_available() == True`,
    `cuda_version=12.8`, `device_count=1`;
  - heavier DINO, cVAE, or full-scale training should run from a GPU-visible
    shell or approved execution mode.
- Phase / progress / composition supervision discussion archived:
  - `docs/agent_qa/2026-06-06-phase-progress-composition-supervision.md`
  - decision: keep old GeoMoCo `u_t` as the primary normalized geometric
    motion-progress anchor, define it narrowly as a motion-phase scaffold rather
    than semantic task progress, and add temporal-alignment, gripper/contact,
    visual-change, object-progress diagnostic, and `SE(3)` composition metrics
    as auxiliary probes rather than immediate replacements.
- Dedicated Geomoco-WM uv environment created and installed:
  - venv: `/home/user/projects/Geomoco-WM/.venv`;
  - project cache: `/home/user/projects/Geomoco-WM/.uv-cache`;
  - install command:
    `UV_CACHE_DIR=/home/user/projects/Geomoco-WM/.uv-cache UV_HTTP_TIMEOUT=300 uv pip install -e '.[dev]'`;
  - key versions: `torch 2.12.0+cu130`, `h5py 3.16.0`, `scipy 1.15.3`,
    `pytest 9.0.3`, `ruff 0.15.16`;
  - default restricted shell still reports `torch.cuda.is_available() == False`.
- Action Semantics Audit and geodesic metrics upgrade completed:
  - code:
    `src/geomoco_wm/data/action_semantics.py`,
    `scripts/audit_libero_action_semantics.py`,
    `src/geomoco_wm/metrics/action_metrics.py`;
  - formal record:
    `docs/experiments/runs/2026-06-06_action_semantics_audit_geodesic_metrics.md`;
  - audit artifacts:
    `outputs/action_semantics/libero_four_suite_action_semantics_audit.json`,
    `outputs/action_semantics/libero_four_suite_action_semantics_audit.md`;
  - result: 4 suites, 40 HDF5 files, 2000 demos, no warnings;
  - readiness: all actions are 7D, all controllers are `OSC_POSE`, all use
    delta control, normalized input range is `[-1, 1]`, output scale is
    `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`;
  - metric semantics: translation is now reported after `0.05m` scaling, and
    rotation is now reported as SO(3) geodesic error between scaled rotvec
    exponentials.
- Verification after geodesic metric upgrade:
  - `.venv/bin/python -m compileall src scripts tests`;
  - `.venv/bin/python -m unittest discover -s tests`;
  - `.venv/bin/ruff check ...`;
  - `git diff --check`;
  - result: all checks passed, including 14 unit tests.
- Geodesic metric write-path smoke completed:
  - command: 1-epoch CPU run on
    `outputs/libero_windows/libero_all_suites_smoke/windows.jsonl`;
  - output:
    `outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta_geodesic_1epoch/metrics.json`;
  - final validation examples:
    `val_translation_m_l2=0.043357`,
    `val_rotation_geodesic_rad=0.036910`,
    `val_rotation_geodesic_deg=2.114782`;
  - interpretation: metric plumbing works; this is not a performance result.
- Gate 1.6 geodesic replacement completed on the same 2-files-per-suite slice:
  - formal record:
    `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_geodesic_replacement.md`;
  - comparison summary:
    `docs/experiments/comparisons/2026-06-06_gate1_6_geodesic_replacement_summary.md`;
  - output metrics:
    `outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed7/metrics.json`,
    `outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/metrics.json`,
    `outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed17/metrics.json`,
    `outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/metrics.json`;
  - mean direct-context validation metrics:
    `val_mse=0.066010`, `val_mae=0.147124`,
    `val_translation_m_l2=0.019024m`,
    `val_rotation_geodesic_deg=2.233651`;
  - mean oracle-future-motion validation metrics:
    `val_mse=0.031474`, `val_mae=0.079508`,
    `val_translation_m_l2=0.007466m`,
    `val_rotation_geodesic_deg=1.048033`;
  - mean reductions from direct context to oracle future motion:
    flat MSE `52.32%`, MAE `45.96%`, translation meter L2 `60.76%`,
    SO(3) geodesic rotation `53.08%`, gripper MSE `26.87%`,
    SE(3) MSE `82.99%`.
- Gate 2 deterministic future-motion prior implemented and run:
  - code:
    `src/geomoco_wm/models/future_motion_predictor.py`,
    `src/geomoco_wm/metrics/motion_metrics.py`,
    `scripts/train_future_motion_predictor.py`;
  - tests:
    `tests/test_future_motion_predictor.py`;
  - formal record:
    `docs/experiments/runs/2026-06-06_gate2_deterministic_future_motion_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-06_gate2_learned_prior_vs_bounds.md`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_deterministic_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_deterministic_seed17/metrics.json`;
  - mean future-motion validation metrics:
    `val_mse=0.001027`, `val_mae=0.019540`,
    `val_l2=0.199044`, `val_translation_l2=0.024785`,
    `val_orientation_coord_l2=0.055226`;
  - mean zero-motion baseline:
    `val_mse=0.001896`, `val_l2=0.250595`,
    `val_translation_l2=0.029872`,
    `val_orientation_coord_l2=0.067285`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.081291`, `action_mae=0.166503`,
    `translation_m_l2=0.022733m`,
    `rotation_geodesic_deg=2.189888`, `gripper_mse=0.296775`;
  - interpretation: learned motion beats zero future motion, but downstream
    action metrics do not beat the direct-context lower bound
    (`action_mse=0.066010`, `translation_m_l2=0.019024m`,
    `rotation_geodesic_deg=2.233651`, `gripper_mse=0.252545`).
  - branch-reading clarification archived in:
    `docs/experiments/runs/2026-06-06_gate2_deterministic_future_motion_prior.md`
    and
    `docs/experiments/comparisons/2026-06-06_gate2_learned_prior_vs_bounds.md`;
    in that clarification, "it" explicitly means the deterministic
    `FutureMotionPredictor(context/proprio) -> predicted future EEF delta`.
- Verification after Gate 2 implementation:
  - `.venv/bin/python -m compileall src scripts tests`;
  - `.venv/bin/python -m unittest discover -s tests`;
  - `.venv/bin/ruff check ...`;
  - `git diff --check`;
  - result: all checks passed, including 16 unit tests.
- Gate 2.1 suite/task-conditioned future-motion prior implemented and run:
  - code:
    `src/geomoco_wm/models/future_motion_predictor.py` now supports optional
    `conditioning_dim`;
    `scripts/train_future_motion_predictor.py` now supports
    `--condition-on none|suite|task|suite_task`;
  - formal record:
    `docs/experiments/runs/2026-06-07_gate2_1_suite_task_conditioned_future_motion_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-07_gate2_1_conditioned_prior_vs_bounds.md`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_1_suite_task_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_1_suite_task_seed17/metrics.json`;
  - conditioning: one-hot `suite_task`, dim 8, built from full dataset
    metadata labels;
  - mean future-motion validation metrics:
    `val_mse=0.000929`, `val_mae=0.018317`,
    `val_l2=0.187877`, `val_translation_l2=0.021964`,
    `val_orientation_coord_l2=0.053070`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.072501`, `action_mae=0.155573`,
    `translation_m_l2=0.020841m`,
    `rotation_geodesic_deg=2.106491`, `gripper_mse=0.279382`;
  - interpretation: suite/task metadata helps compared with Gate 2
    (`action_mse=0.081291` -> `0.072501`), but still does not beat direct
    context (`action_mse=0.066010`), so it remains a diagnostic baseline.
- Gate 2.2a DINOv2 global visual future-motion prior implemented and run:
  - code:
    `src/geomoco_wm/data/visual_feature_cache.py`,
    `scripts/cache_libero_dino_features.py`;
    `src/geomoco_wm/data/window_dataset.py` now supports optional visual
    feature cache attachment;
    `scripts/train_future_motion_predictor.py` now supports
    `--visual-feature-cache`;
  - discussion archive:
    `docs/agent_qa/2026-06-07-visual-grounding-design-and-gate22-plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-07_gate2_2a_dinov2_global_visual_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-07_gate2_2a_visual_prior_vs_bounds.md`;
  - visual cache:
    `outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5`,
    dim 1536 from `dinov2_vits14_reg`, 2 context frames, and two cameras;
  - artifacts:
    `outputs/future_motion_predictor/gate2_2a_dinov2_global_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_2a_dinov2_global_seed17/metrics.json`;
  - mean future-motion validation metrics:
    `val_mse=0.000801`, `val_mae=0.016178`,
    `val_l2=0.171171`, `val_translation_l2=0.016219`,
    `val_orientation_coord_l2=0.050310`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.053628`, `action_mae=0.128207`,
    `translation_m_l2=0.016285m`,
    `rotation_geodesic_deg=2.031589`, `gripper_mse=0.229947`;
  - interpretation: visual grounding produces the first learned prior that
    beats direct context (`action_mse=0.053628` vs `0.066010`) and closes
    `35.85%` of the direct-to-oracle action-MSE gap.
- Gate 2.2b patch-pooled DINOv2 cross-attention visual prior implemented and
  run:
  - code:
    `scripts/cache_libero_dino_features.py` now supports
    `--feature-mode patch_pool`;
    `src/geomoco_wm/models/future_motion_predictor.py` now includes
    `VisualCrossAttentionFutureMotionPredictor`;
    `scripts/train_future_motion_predictor.py` now supports
    `--visual-fusion cross_attention`;
  - formal record:
    `docs/experiments/runs/2026-06-07_gate2_2b_patchpool_cross_attention_visual_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-07_gate2_2_visual_grounding_summary.md`;
  - visual cache:
    `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5`,
    64 visual tokens per window, token dim 384, flat dim 24576;
  - artifacts:
    `outputs/future_motion_predictor/gate2_2b_patchpool4_crossattn_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_2b_patchpool4_crossattn_seed17/metrics.json`;
  - mean future-motion validation metrics:
    `val_mse=0.000772`, `val_mae=0.015710`,
    `val_l2=0.168029`, `val_translation_l2=0.014441`,
    `val_orientation_coord_l2=0.050114`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.049547`, `action_mae=0.120370`,
    `translation_m_l2=0.014859m`,
    `rotation_geodesic_deg=2.030450`, `gripper_mse=0.222467`;
  - interpretation: patch cross-attention improves action MSE by `7.61%`
    over Gate 2.2a and closes `47.67%` of the direct-to-oracle action-MSE gap.

## Current Interpretation
- The repository is ready as a clean visual-grounded method track and handoff
  point.
- The current code proves only package structure and minimal tensor plumbing.
  It does not yet prove visual feature export, grounding quality, cVAE training
  stability, predictive value, closed-loop performance, or baseline fairness.
- The project should not claim full world-model status until it predicts or
  samples future motion from visual/proprio/task context and shows value in
  predictive or controlled action-decoder gates.
- Old GeoMoCo evidence should be treated as a warning signal: DINO grounding and
  candidate retrieval may improve representation recall without producing
  closed-loop gains if progress precision, phase estimation, or action execution
  is still the real bottleneck.
- The new blueprint turns that warning into explicit gates: task mining,
  visual-grounding probes, oracle future-motion action decoding, cVAE coverage,
  shared decoder attribution, mechanism closed-loop, and only then
  LIBERO-Long/LIBERO-10 main evaluation.
- The first implementation step is intentionally read-only. It validates HDF5
  field availability and length alignment before committing to an exporter
  schema, which keeps data-side mistakes from leaking into DINO/cache/training.
- Gate 0 passed for `libero_goal`. The first exporter can require
  `agentview_rgb`, `eye_in_hand_rgb`, `ee_states` or `ee_pos + ee_ori`,
  `gripper_states`, `joint_states`, and 7D `actions`. Object-state should not be
  required as a normal model input.
- Gate 1 defines the first executable data contract. For anchor timestep `t`,
  context frames are `[t - context_len + 1, ..., t]`, future EEF target frames
  are `[t + 1, ..., t + horizon]`, and action chunks are
  `[t, ..., t + horizon - 1]`.
- The current exporter materializes numeric targets in JSONL for readability.
  If full-suite JSONL becomes too large, switch targets to HDF5/NPZ while
  preserving the same window index semantics.
- The first experiment script now exists and maps GT future EEF deltas plus
  proprioceptive context to action chunks. This directly serves the oracle
  action-decoder gate before any cVAE or DINO training.
- The first 10-demo oracle smoke is positive: GT future EEF deltas help the MLP
  decoder over direct context on the drawer subset. This supports continuing
  the executable-interface gate, but it is not yet enough to start cVAE claims.
- Because all four standard LIBERO suites are now local, the oracle-vs-direct
  diagnostic should be promoted from single-task/window-level smoke to
  multi-suite episode-level comparison before cVAE or DINO claims.
- The first interpretable four-suite test should be the small formal slice:
  one HDF5 task file per suite, all demos in those files, horizon 8, stride 4,
  and episode-level train/validation split. Smoke results remain plumbing-only.
- Gate 1.5 is now positive: across two episode-level seeds, GT future EEF
  deltas reduced validation MSE by roughly 51-57% and MAE by roughly 43-47%.
  This supports continuing toward learned future-motion priors, while still not
  proving visual grounding, cVAE sampling, or closed-loop success.
- Gate 1.6 confirms the mechanism at a larger slice: two files per suite still
  gives roughly 51-54% flat MSE reduction and 82-84% SE(3) MSE reduction. The
  oracle future-motion interface is robust enough to promote a learned
  future-motion prior as the next mainline experiment.
- The action metric contract is now confirmed against LIBERO HDF5 metadata and
  local robosuite source behavior. Future action-decoder runs should report
  `translation_m_*`, `rotation_rotvec_rad_*`, and `rotation_geodesic_*` in
  addition to normalized MSE/MAE so results are physically interpretable.
- The Gate 1.6 replacement makes the oracle gap physically readable: the
  mean validation translation action error drops from `0.0190m` to `0.00747m`,
  and the mean SO(3) geodesic rotation error drops from `2.23deg` to `1.05deg`.
  This is the clean reference table for the first learned future-motion prior.
- Gate 2 shows a useful failure mode: context-only deterministic future-motion
  prediction reduces future-delta MSE compared with zero motion, but its
  predicted motion does not improve the downstream action decoder over direct
  context. This suggests the next prior needs stronger conditioning and/or an
  action-aware objective, not merely lower future-motion MSE.
- Gate 2.1 narrows but does not solve that failure mode: task/suite metadata
  reduces future-motion MSE and downstream action MSE, yet the action route is
  still worse than direct context. This suggests the missing signal is not only
  task identity; the prior likely needs visual grounding, temporal structure,
  gripper/contact modeling, or action-aware supervision.
- Gate 2.2a gives the first positive visual-grounding mechanism result:
  DINOv2 global visual features make predicted future EEF motion executable
  enough to beat direct context through the frozen action-decoder interface.
  This supports the GeoMoCo-WM thesis that vision is not merely an auxiliary
  modality, but a grounding signal that turns future motion into a more useful
  policy intermediate variable.
- Gate 2.2b strengthens that result: patch-pooled cross-attention is the best
  learned prior so far, but the gain over global DINO is moderate. This makes
  visual controls mandatory before cVAE: shuffled features should fail, and
  camera ablations should reveal which view carries the value.
- The remaining gap from Gate 2.2b to oracle future motion is expected because
  oracle future motion is privileged future trajectory information. The gap is
  now a research target, not a negative result: first verify visual attribution
  with controls, then use action-aware or multimodal priors to reduce the
  remaining direct-to-oracle gap.
- The metric decomposition suggests a split modeling plan: future EEF motion is
  the primary geometric branch, while gripper/contact should be modeled or
  supervised separately because its gains are positive but much smaller.
- Future formal experiment results should be recorded under
  `docs/experiments/runs/`, with cross-run ablations and promotion decisions
  under `docs/experiments/comparisons/`.
- Phase, progress, and composition should be reported separately. `u_t` remains
  the lowest-cost stable coordinate for segment construction and motion-phase
  probing, while semantic progress should be treated as task-specific and
  diagnostic unless clean object/contact/success labels exist.
- GeoMoCo-WM should evaluate new progress heuristics by mechanism value, not by
  reconstruction alone: future state/progress prediction, composition
  hard-negative ranking, future-motion coverage, and the motion-to-action
  decoder gap are the promotion gates.

## Open Decisions Or Blockers
- Run heavy DINO, cVAE, or large full-suite training from a GPU-visible shell or
  approved execution mode; the default restricted context hides CUDA even
  though elevated Python can see the 5090.
- Decide whether JSONL target materialization is acceptable for the next reader,
  or whether to move numeric arrays to HDF5/NPZ before full export.
- Full four-suite export remains optional; the next mainline experiment can
  start with the 2-files-per-suite slice because the oracle interface already
  scaled positively.
- Treat object-state teacher fields as unavailable for `libero_goal`; only add
  object-state upper bounds if a later suite/source provides them.
- Define the exact visual-grounded dataset contract, including RGB history,
  DINO feature cache, proprioception, EEF pose, gripper, task, future motion,
  action chunk, and optional object-state teacher fields.
- Task/suite conditioning alone was tested and is not sufficient; DINO global
  and patch visual grounding are positive, but still need shuffled-vision and
  camera controls before cVAE claims.
- Next-step sequencing decision archived in
  `docs/agent_qa/2026-06-08-oracle-gap-and-next-step-sequencing.md`: visual
  controls come first, then action-aware/multimodal gap reduction if controls
  pass.
- Define the first `u_geom` target builder for exported LIBERO windows and
  decide whether phase bins / temporal-rank pairs are generated during export
  or in a separate target-materialization pass.
- Choose DINO backbone/version and feature-cache format.
- Decide the first action decoder: simple action-chunk transformer first, with
  Diffusion Policy / MeanFlow-style decoder only after attribution is clear.
- The small formal oracle future-motion diagnostic passed; if a larger-scale
  run fails to beat direct BC, redesign the interface or task suite before
  spending more effort on the cVAE-to-action route.
- Decide the concrete LIBERO-Long/LIBERO-10 task subset after a headroom audit:
  avoid tasks where direct DINO/BC is saturated, and avoid tasks where all
  policies fail for unrelated execution reasons.
- Decide whether object state is teacher/diagnostic only or included in any
  upper-bound baseline.
- Project dev dependencies are installed in the dedicated `.venv`; use
  `.venv/bin/python -m unittest discover -s tests` and `.venv/bin/ruff check`
  for local checks.

## Next Session Entry Point
1. Run Gate 2.2 controls: shuffled DINO features and agentview-only /
   eye-in-hand-only / two-camera camera ablations.
2. If controls pass, add a stronger temporal predictor or action-aware
   auxiliary loss after
   visual controls identify whether global or patch grounding carries the value.
3. Then evaluate multimodal future-motion priors, such as cVAE / diffusion /
   flow, to reduce the remaining oracle gap.
4. Add a separate gripper/contact auxiliary target or diagnostic because Gate
   1.6 shows geometric action improves much more than gripper action.
5. For heavy training, use a GPU-visible shell or approved execution mode; the
   default restricted context sees `torch.cuda.is_available() == False`, while
   elevated Python sees the RTX 5090 correctly.
6. Add Gate-2 visual grounding probes for future EEF SE(3), geometric progress,
   and optional future DINO feature prediction.
7. Add `u_geom`, phase-bin, temporal-rank, and composition hard-negative
   diagnostics before treating any learned progress head as semantic.
8. Extend GeoMoCo-cVAE context with visual grounding token `g_t` and add
   deterministic AE, stochastic cVAE, direct residual, random, shuffled, and
   oracle future-motion baselines.
9. Before large cVAE training, compare direct BC with a controlled
   `GT future motion -> action decoder` upper bound.
10. Build the task-mining/headroom audit so mechanism tasks and LIBERO-Long /
   LIBERO-10 tasks are selected before large model training.
