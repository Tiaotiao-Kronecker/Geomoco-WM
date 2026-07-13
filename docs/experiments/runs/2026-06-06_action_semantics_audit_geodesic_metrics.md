# Action Semantics Audit And Geodesic Metrics Upgrade

Date: 2026-06-06

## Purpose

Confirm the exact LIBERO action semantics before promoting action metrics from
normalized split MSE/MAE to physical translation and SO(3) geodesic rotation
metrics.

This is a measurement-contract run, not a model-performance run.

## Environment

- project env: `/home/user/projects/Geomoco-WM/.venv`
- project uv cache: `/home/user/projects/Geomoco-WM/.uv-cache`
- install command:

```bash
UV_CACHE_DIR=/home/user/projects/Geomoco-WM/.uv-cache UV_HTTP_TIMEOUT=300 uv pip install -e '.[dev]'
```

- verification import:
  - Python: `/home/user/projects/Geomoco-WM/.venv/bin/python`
  - `torch`: `2.12.0+cu130`
  - `h5py`: `3.16.0`
  - `scipy`: `1.15.3`
  - default restricted shell CUDA visibility: `torch.cuda.is_available() == False`

## Code Changes

- `src/geomoco_wm/data/action_semantics.py`
  - canonical LIBERO `OSC_POSE` action semantics;
  - HDF5 metadata audit for controller type, delta mode, input range, output
    scale, and action dimension.
- `scripts/audit_libero_action_semantics.py`
  - CLI for single-suite or four-suite action-semantics audit.
- `src/geomoco_wm/metrics/action_metrics.py`
  - normalized flat and split action metrics;
  - physical translation metrics after `0.05m` scaling;
  - scaled rotvec metrics after `0.5rad` scaling;
  - SO(3) geodesic metric between `Exp(pred_rotvec_scaled)` and
    `Exp(target_rotvec_scaled)`.
- `scripts/train_oracle_action_decoder.py`
  - now records `action_semantics` in `metrics.json`;
  - now emits `translation_m_*`, `rotation_rotvec_rad_*`, and
    `rotation_geodesic_*` metrics.
- tests:
  - `tests/test_action_semantics_audit.py`
  - `tests/test_oracle_action_decoder_metrics.py`

## Audit Command

```bash
.venv/bin/python scripts/audit_libero_action_semantics.py \
  --all-libero-suites \
  --dataset-root /home/user/dataset/libero_official \
  --output-json outputs/action_semantics/libero_four_suite_action_semantics_audit.json \
  --output-md outputs/action_semantics/libero_four_suite_action_semantics_audit.md
```

## Audit Result

- suites: `4`
- HDF5 files: `40`
- demos: `2000`
- warnings: none

Readiness:

- `supports_geodesic_action_metrics`: `true`
- `all_actions_7d`: `true`
- `all_controllers_osc_pose`: `true`
- `all_control_delta`: `true`
- `all_input_range_matches`: `true`
- `all_output_scale_matches`: `true`

Canonical action semantics:

- action dim: `7`
- translation action dims: `0:3`, normalized delta scaled by `0.05m`
- rotation action dims: `3:6`, normalized rotvec scaled by `0.5rad`
- gripper dim: `6`
- controller rotation rule:
  `R_goal = Exp(rotvec_scaled) @ R_current`

Artifacts:

- JSON:
  `outputs/action_semantics/libero_four_suite_action_semantics_audit.json`
- Markdown:
  `outputs/action_semantics/libero_four_suite_action_semantics_audit.md`

## Metric Definitions

For predicted action chunk `a_hat` and target action chunk `a`, with translation
scale `s_t = (0.05, 0.05, 0.05)` and rotation scale
`s_r = (0.5, 0.5, 0.5)`:

```text
translation_m_error = (a_hat[..., 0:3] - a[..., 0:3]) * s_t
translation_m_l2 = mean_t ||translation_m_error_t||_2

pred_rotvec = a_hat[..., 3:6] * s_r
target_rotvec = a[..., 3:6] * s_r
rotation_geodesic_rad =
  mean_t acos((trace(Exp(pred_rotvec_t)^T Exp(target_rotvec_t)) - 1) / 2)
```

Flat normalized MSE/MAE and normalized translation/rotation/gripper split
metrics are kept for backward comparison with Gate 1.5 and Gate 1.6.

## Smoke Command

```bash
.venv/bin/python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_smoke/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta_geodesic_1epoch \
  --epochs 1 \
  --batch-size 4 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 7 \
  --device cpu
```

Smoke result:

- metrics JSON:
  `outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta_geodesic_1epoch/metrics.json`
- final validation examples:
  - `val_translation_m_l2`: `0.043357`
  - `val_rotation_geodesic_rad`: `0.036910`
  - `val_rotation_geodesic_deg`: `2.114782`

Interpretation: the metric-write path works. This smoke is not a performance
claim because it uses only the tiny four-suite plumbing slice and one epoch.

## Verification

```bash
.venv/bin/python -m compileall src scripts tests
.venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check src/geomoco_wm/data/action_semantics.py src/geomoco_wm/metrics/action_metrics.py scripts/audit_libero_action_semantics.py scripts/train_oracle_action_decoder.py tests/test_action_semantics_audit.py tests/test_oracle_action_decoder_metrics.py
git diff --check
```

Results:

- `compileall`: passed
- `unittest`: 14 tests passed
- `ruff`: passed
- `git diff --check`: passed

## Interpretation

The previous wording "SE(3)-aware split metrics" is now upgraded:

- old: normalized action dimensions split into translation, rotation, `se3`,
  and gripper groups;
- new: metric semantics are tied to LIBERO controller metadata and local
  robosuite source behavior;
- new: translation is reported in meters;
- new: rotation is reported as SO(3) geodesic radians/degrees.

This is the stable route: we do not change the model, dataset, or training
objective before locking the measurement contract.

## Next Decision

Rerun the 2-files-per-suite oracle action-decoder gate with the new metrics, or
move directly to the learned future-motion prior while reporting the new
metrics in all future action-decoder runs.
