# Episode Split And Four-Suite Slice

- Date: 2026-06-06
- Project: `Geomoco-WM`
- Topic: why the oracle action-decoder gate should move from `libero_goal`
  smoke to four-suite episode-level evaluation

## Question

After expanding local LIBERO data from only `libero_goal` to the standard four
suites, should the `direct context` vs `GT future motion` oracle action-decoder
comparison also be expanded?

Short answer: yes. The previous positive result on a 10-demo `libero_goal`
drawer subset is useful, but it is not enough evidence for the GeoMoCo-WM route.
The diagnostic should next run on a small but real four-suite slice with
episode-level train/validation splits.

## Episode-Level Split

The exporter turns each full LIBERO demonstration into many overlapping sliding
windows. For one demo:

```text
demo_000
  window 1: frames 0-17
  window 2: frames 4-21
  window 3: frames 8-25
  window 4: frames 12-29
```

If train/validation split is done at the window level, the validation set can
contain windows that overlap heavily with training windows from the same demo:

```text
train: window 1, window 3, window 4
val:   window 2
```

This leaks near-identical temporal context into validation. A good validation
score may then reflect local trajectory memorization instead of useful
generalization.

Episode-level split groups by full demo:

```text
train: all windows from demo_000, demo_001, demo_002, ...
val:   all windows from demo_010, demo_011, ...
```

The rule is:

```text
all windows from the same episode/demo must go entirely to train or entirely to val
```

This is the right split for the four-suite oracle gate because it asks whether
future-motion information helps on unseen demonstrations, not merely adjacent
windows.

## Direct Context Vs GT Future Motion

The current MLP diagnostic compares two inputs to the same action decoder:

```text
direct context:
  anchor EEF + current gripper + current joint
  -> action chunk

GT future motion:
  anchor EEF + current gripper + current joint
  + true future EEF deltas
  -> action chunk
```

In code this is controlled by:

```text
--motion-mode none
--motion-mode future_delta
```

`future_delta` is an oracle upper bound: it uses ground-truth future EEF motion.
If this oracle does not reliably beat direct context, then a learned GeoMoCo
future-motion latent is unlikely to help the action policy without redesigning
the representation, action head, or task setup.

## Four-Suite Small Formal Slice

The proposed next slice is:

```text
run four suite exporter
  suite_names = libero_spatial, libero_object, libero_goal, libero_10
  max_files_per_suite = 1
  max_demos_per_file = unset
  max_windows_per_suite = unset
```

Meaning:

- choose one HDF5 task file from each suite;
- use all demos inside that file, usually about 50 demonstrations per selected
  task;
- export all valid sliding windows from those demos;
- combine the four suite outputs into one `windows.jsonl`;
- train direct-context and GT-future-motion decoders on the same combined
  windows;
- split train/val by episode;
- repeat with at least two seeds.

This is "small" because it uses only one task file per suite instead of all
40 task files. It is "formal" because it uses all demos in selected files,
episode-level validation, and repeated seeds.

## Suggested Commands

First export one file per suite using all demos:

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

Then run the direct-context baseline:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed7 \
  --motion-mode none \
  --split-by episode \
  --seed 7
```

Run the oracle future-motion branch:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_future_seed7 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 7
```

Repeat both with another seed, for example `--seed 17`.

## Interpretation

Pass signal:

```text
GT future motion beats direct context on validation MSE/MAE across seeds.
```

This means future geometric motion contains action-useful information under a
less leaky split and across multiple suite types.

Weak or failed signal:

```text
GT future motion does not beat direct context consistently.
```

Then the immediate priority should not be cVAE scale-up. Instead, revisit one of
these:

- future-motion representation;
- action head capacity;
- action metric, especially SE(3)-aware translation/rotation/gripper reporting;
- task slice selection;
- visual grounding and phase/progress precision.

## Why Not Full Four-Suite Export First

The full four-suite export is now technically possible, but the next step should
check:

- JSONL size;
- training-loop speed;
- suite/task balance;
- whether the oracle gap survives episode-level validation.

After that, run the full four-suite export and promote the diagnostic to the
main gate.
