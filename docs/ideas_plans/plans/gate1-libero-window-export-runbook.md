# Gate 1 LIBERO Window Export Runbook

- Date: 2026-06-05
- Status: first exporter implemented, tiny real-data smoke passed
- Scope: HDF5 references plus lightweight numeric future-motion/action windows

## Purpose

Gate 1 converts official LIBERO HDF5 demos into the first GeoMoCo-WM dataset
contract:

```text
official LIBERO HDF5
  -> episodes.jsonl
  -> windows.jsonl
  -> summary.json
```

This exporter still does not save RGB frames and does not run DINO. RGB remains
in the original HDF5 files and windows store frame indices plus camera dataset
references. The materialized numeric targets are small enough to inspect:

- anchor EEF state;
- future EEF states;
- future coordinate deltas from anchor EEF state;
- future action chunk;
- current gripper and joint state.

## Window Semantics

For anchor timestep `t`:

```text
context frames: [t - context_len + 1, ..., t]
future EEF frames: [t + 1, ..., t + horizon]
action chunk: [t, ..., t + horizon - 1]
```

The action chunk is aligned with the future EEF target horizon. This is the
first data contract needed for the oracle diagnostic:

```text
GT future motion -> action decoder
```

## New Artifacts

```text
src/geomoco_wm/data/libero_hdf5_export.py
scripts/export_libero_windows.py
tests/test_libero_hdf5_export.py
```

## Local Verification Already Run

Synthetic tests:

```bash
python -m compileall src scripts tests
python -m unittest discover -s tests -p 'test_libero_hdf5_*.py'
```

Result:

```text
7 tests passed
```

CLI check:

```bash
python scripts/export_libero_windows.py --help
```

## Tiny Real-Data Smoke

Command:

```bash
python scripts/export_libero_windows.py \
  --input-path /home/user/dataset/libero_official/libero_goal \
  --suite-name libero_goal \
  --output-dir outputs/libero_windows/libero_goal_smoke \
  --context-len 2 \
  --horizon 8 \
  --stride 4 \
  --max-files 1 \
  --max-demos-per-file 1 \
  --max-windows 3
```

Result:

```text
episodes.jsonl: 1 record
windows.jsonl: 3 records
summary.json: 1 file, 1 demo, 138 frames
```

Output:

```text
outputs/libero_windows/libero_goal_smoke/episodes.jsonl
outputs/libero_windows/libero_goal_smoke/windows.jsonl
outputs/libero_windows/libero_goal_smoke/summary.json
```

The expected warning is:

```text
Export stopped early because max_windows was reached.
```

This warning is correct for the smoke run.

## Four-Suite Batch Smoke

After downloading the standard four LIBERO suites to:

```text
/home/user/dataset/libero_official
```

the exporter can run in batch mode:

```bash
python scripts/export_libero_windows.py \
  --all-libero-suites \
  --input-root /home/user/dataset/libero_official \
  --output-dir outputs/libero_windows/libero_all_suites_smoke \
  --context-len 2 \
  --horizon 8 \
  --stride 4 \
  --max-files-per-suite 1 \
  --max-demos-per-file 1 \
  --max-windows-per-suite 2
```

Result:

```text
libero_spatial: 2 windows
libero_object: 2 windows
libero_goal: 2 windows
libero_10: 2 windows
combined windows.jsonl: 8 records
combined episodes.jsonl: 4 records
```

The batch output keeps each suite's files under its own subdirectory and also
writes combined files:

```text
outputs/libero_windows/libero_all_suites_smoke/{suite}/windows.jsonl
outputs/libero_windows/libero_all_suites_smoke/windows.jsonl
```

## Review Checklist

All relative paths in this runbook assume the current working directory is:

```text
/home/user/projects/Geomoco-WM
```

From `/home/user/projects`, prefix paths with `Geomoco-WM/`.

Inspect the first window:

```bash
sed -n '1p' outputs/libero_windows/libero_goal_smoke/windows.jsonl | python -m json.tool
```

Check:

- `context_frame_indices` uses the requested context history;
- `anchor_index` is the final context frame;
- `future_frame_indices` starts at `anchor_index + 1`;
- `action_start == anchor_index`;
- `action_end - action_start == future_end - future_start`;
- `future_delta_ee_states` is `future_ee_states - anchor_ee_state`;
- `camera_keys` includes `agentview_rgb` and `eye_in_hand_rgb`.

## Next Safe Export

After reviewing the smoke output, run a slightly larger export without hitting
the full 500-demo suite:

```bash
python scripts/export_libero_windows.py \
  --input-path /home/user/dataset/libero_official/libero_goal \
  --suite-name libero_goal \
  --output-dir outputs/libero_windows/libero_goal_task0_10demo \
  --context-len 2 \
  --horizon 16 \
  --stride 4 \
  --max-files 1 \
  --max-demos-per-file 10
```

Only after that should we decide whether to run the full suite export.

## Stop Rules

Stop before larger export if:

- the window alignment rule is wrong for the policy/action convention;
- EEF `ee_states` are not the target representation we want for the first
  oracle action-decoder diagnostic;
- storing numeric target arrays in JSONL becomes too large or too slow;
- we decide the first training dataloader should read all targets lazily from
  HDF5 rather than materializing them in `windows.jsonl`.

## Next Gate

Gate 2 should add one small dataset reader over `windows.jsonl`, then run
sanity probes before DINO:

```text
window JSONL
  -> torch Dataset
  -> oracle future EEF/action decoder smoke
  -> only then DINO cache and visual grounding
```

## Experiment Script Added

The first experiment-facing script is now:

```text
scripts/train_oracle_action_decoder.py
```

It trains the diagnostic:

```text
anchor EEF + gripper + joint context
+ GT future EEF delta
  -> action chunk
```

Dry-run shape check on the tiny smoke export:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_goal_smoke/windows.jsonl \
  --dry-run
```

Tiny CPU smoke:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_goal_smoke/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_goal_smoke \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --hidden-dims 32,32
```

This smoke only proves the training loop runs. It is not a meaningful metric
because the input has only 3 windows.

For four-suite direct-context vs oracle-future-motion checks, use the combined
batch `windows.jsonl` and prefer episode-level splits:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_smoke/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta \
  --motion-mode future_delta \
  --split-by episode \
  --dry-run

python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_smoke/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_smoke_direct_context \
  --motion-mode none \
  --split-by episode \
  --dry-run
```

The dry-run should report suite counts for all four suites and `motion_dim=48`
for `future_delta` versus `motion_dim=0` for `none` when `horizon=8`.
