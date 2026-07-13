# Four-Suite Test Dataset Usage

- Date: 2026-06-06
- Project: `Geomoco-WM`
- Topic: how to use the local four-suite LIBERO test data for the next
  oracle action-decoder gate

## Local Dataset

The local official LIBERO HDF5 root is:

```text
/home/user/dataset/libero_official
```

The standard four suites are present:

```text
/home/user/dataset/libero_official/libero_spatial
/home/user/dataset/libero_official/libero_object
/home/user/dataset/libero_official/libero_goal
/home/user/dataset/libero_official/libero_10
```

Each suite has 10 HDF5 task files. The current validated local state is:

```text
libero_spatial: 10 files, hdf5_ok=10
libero_object: 10 files, hdf5_ok=10
libero_goal: 10 files, hdf5_ok=10
libero_10: 10 files, hdf5_ok=10
```

## Dataset Use Policy

Use three dataset scales:

```text
smoke:
  max-files-per-suite = 1
  max-demos-per-file = 1
  max-windows-per-suite = small cap

small formal slice:
  max-files-per-suite = 1
  max-demos-per-file = unset
  max-windows-per-suite = unset

full four-suite:
  max-files-per-suite = unset
  max-demos-per-file = unset
  max-windows-per-suite = unset
```

The smoke scale checks plumbing only. It should not be interpreted as a
performance result.

The small formal slice is the next real diagnostic. It uses one task file from
each suite and all demonstrations in those selected task files.

The full four-suite export should wait until JSONL size, training speed, and
the oracle-vs-direct gap are acceptable on the small formal slice.

## Small Formal Slice Export

Command:

```bash
python scripts/export_libero_windows.py \
  --all-libero-suites \
  --input-root /home/user/dataset/libero_official \
  --output-dir outputs/libero_windows/libero_all_suites_1file_all_demos_h8 \
  --context-len 2 \
  --horizon 8 \
  --stride 4 \
  --max-files-per-suite 1
```

Important details:

- `--all-libero-suites` selects `libero_spatial`, `libero_object`,
  `libero_goal`, and `libero_10`.
- `--max-files-per-suite 1` selects one HDF5 task file from each suite.
- no `--max-demos-per-file` means use all demos inside each selected task file.
- no `--max-windows-per-suite` means use all valid sliding windows from those
  demos.
- the exporter writes both per-suite JSONL and combined JSONL.

Combined output:

```text
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/episodes.jsonl
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/summary.json
```

## Oracle Action-Decoder Comparison

The comparison has two branches using the same dataset and decoder:

```text
direct context:
  --motion-mode none

GT future motion:
  --motion-mode future_delta
```

Use episode-level split:

```text
--split-by episode
```

This prevents adjacent windows from the same demonstration leaking across
train/validation.

Run at least two seeds:

```text
seed 7
seed 17
```

Suggested commands:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed7 \
  --motion-mode none \
  --split-by episode \
  --seed 7

python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_future_seed7 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 7
```

Repeat with `--seed 17`.

## Interpretation

Pass signal:

```text
GT future motion consistently improves validation MSE/MAE over direct context.
```

This supports continuing toward learned future-motion priors and then
visual-grounded GeoMoCo-WM.

Stop or redesign signal:

```text
GT future motion does not consistently improve over direct context.
```

Then the priority is not larger cVAE/DINO training. Revisit action head
capacity, SE(3)-aware metrics, motion representation, task selection, and
phase/progress grounding.
