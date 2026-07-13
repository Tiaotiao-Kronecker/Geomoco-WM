# Gate 0 LIBERO HDF5 Inspection Runbook

- Date: 2026-06-05
- Status: code written, real LIBERO inspection pending user review
- Scope: read-only dataset structure audit before any image export, DINO cache,
  window build, or training job

## Purpose

Gate 0 should answer one narrow question:

```text
Do the local official LIBERO HDF5 files expose the fields needed to build the
first visual-grounded GeoMoCo-WM dataset?
```

This gate deliberately avoids all heavy work. It does not save RGB frames, does
not run DINO, does not build motion windows, and does not touch GPU. It only
opens HDF5 metadata and reports keys, shapes, per-demo lengths, coverage, and
readiness flags.

## New Artifacts

```text
src/geomoco_wm/data/libero_hdf5_inspect.py
scripts/inspect_libero_hdf5.py
tests/test_libero_hdf5_inspect.py
```

The inspector checks:

- demo-level keys such as `actions`, `obs`, `states`, `robot_states`, `dones`,
  and `rewards`;
- observation keys such as `agentview_rgb`, `eye_in_hand_rgb`, `ee_pos`,
  `ee_ori`, `ee_states`, `gripper_states`, and `joint_states`;
- sequence length alignment across actions, RGB, EEF, gripper, joints, and
  simulator state fields;
- whether action chunks can use 7D actions;
- whether EEF motion targets can be recovered from `ee_states` or
  `ee_pos + ee_ori`;
- whether dual-camera visual grounding is available;
- whether object-state teacher fields are present or must remain
  diagnostic-only.

## Local Verification Already Run

These checks use only synthetic HDF5 files created inside temporary directories:

```bash
python -m compileall src scripts tests
python -m unittest discover -s tests -p 'test_libero_hdf5_inspect.py'
```

Result:

```text
compileall: pass
unittest: 4 tests passed
```

`pytest` is still optional; the current lightweight path uses standard
`unittest` because the active environment previously did not have `pytest`
installed.

## Review Before Real Execution

Read these files first:

```text
src/geomoco_wm/data/libero_hdf5_inspect.py
scripts/inspect_libero_hdf5.py
tests/test_libero_hdf5_inspect.py
```

The important function is:

```python
inspect_libero_hdf5_suite(...)
```

The important readiness flag is:

```text
report["readiness"]["supports_gate0_dataset_export"]
```

Do not start the real export step unless that flag is true, or unless we
explicitly decide that a missing field is optional for the first dataset.

## First Real Smoke Command

After review, run only one file and two demos first:

```bash
python scripts/inspect_libero_hdf5.py \
  --input-path /home/user/dataset/libero_official/libero_goal \
  --suite-name libero_goal \
  --max-files 1 \
  --max-demos-per-file 2 \
  --output-json outputs/gate0/libero_goal_hdf5_inspection_smoke.json \
  --output-md outputs/gate0/libero_goal_hdf5_inspection_smoke.md
```

Inspect:

```text
outputs/gate0/libero_goal_hdf5_inspection_smoke.md
```

The smoke report should be checked for:

- `supports_visual_grounding_export`;
- `supports_dual_camera_export`;
- `supports_eef_motion_targets`;
- `supports_action_chunks`;
- `supports_proprio_context`;
- warnings about object-state teacher fields.

## Full Suite Inspection Command

If the smoke report is clean, run the full `libero_goal` suite:

```bash
python scripts/inspect_libero_hdf5.py \
  --input-path /home/user/dataset/libero_official/libero_goal \
  --suite-name libero_goal \
  --output-json outputs/gate0/libero_goal_hdf5_inspection.json \
  --output-md outputs/gate0/libero_goal_hdf5_inspection.md
```

Expected local scale:

```text
10 HDF5 files
approximately 50 demos per task file
approximately 500 demos total
```

This is still a metadata inspection, not image export.

## Pass Criteria

Gate 0 passes for the first exporter if:

- all scanned demos have aligned sequence lengths;
- all scanned demos have `actions` with final dimension `7`;
- all scanned demos have `agentview_rgb`;
- wrist camera is either universally available, or we decide the first exporter
  can run with agentview-only plus a clearly marked ablation;
- all scanned demos have EEF motion fields from `ee_states` or `ee_pos + ee_ori`;
- gripper and joint states are sufficiently available for proprio-conditioned
  context;
- object-state absence is treated as diagnostic-only, not as a required input.

## Stop Rules

Stop before export if:

- action length and RGB length disagree;
- EEF fields are missing or not 6D recoverable;
- action dimension is not consistently 7D;
- the camera field names differ from the default and need an adapter decision;
- a readiness warning affects a field we planned to use as model input.

## Next Gate After Pass

Gate 0 only approves implementation of the real dataset exporter. The next
implementation slice should build:

```text
official LIBERO HDF5
  -> episode manifest
  -> optional RGB frame export
  -> future-motion/action window index
  -> DINO cache path later
```

Training remains blocked until the exporter and a small dataset audit pass.
