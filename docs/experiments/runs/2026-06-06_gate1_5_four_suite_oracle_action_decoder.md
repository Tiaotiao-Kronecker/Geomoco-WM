# Gate 1.5 Four-Suite Oracle Action Decoder

- Date: 2026-06-06
- Status: completed
- Gate: `Gate 1.5`
- Purpose: test whether future EEF motion provides actionable signal beyond
  direct context/proprioception.

## Dataset Slice

Source:

```text
/home/user/dataset/libero_official
```

Export output:

```text
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/
```

Export command:

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

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 files | 4 |
| demos / episodes | 200 |
| windows | 7,921 |
| frames | 33,201 |
| dropped short episodes | 0 |
| exporter warnings | 0 |
| combined `windows.jsonl` size | 33M |

Per-suite windows:

| suite | episodes | windows |
| --- | ---: | ---: |
| `libero_spatial` | 50 | 1,169 |
| `libero_object` | 50 | 1,857 |
| `libero_goal` | 50 | 1,663 |
| `libero_10` | 50 | 3,232 |

## Model And Training Config

Common settings:

```text
script: scripts/train_oracle_action_decoder.py
model: MLP ActionDecoder
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
split-by: episode
seeds: 7, 17
```

Branches:

| branch | command mode | context dim | motion dim | action dim | horizon |
| --- | --- | ---: | ---: | ---: | ---: |
| direct context | `--motion-mode none` | 15 | 0 | 7 | 8 |
| oracle future motion | `--motion-mode future_delta` | 15 | 48 | 7 | 8 |

The run executed in the default restricted context, where CUDA was hidden, so
the four MLP runs used CPU. Elevated checks confirmed the same Python/Torch
environment can see the RTX 5090 when run with normal GPU access.

## Commands

Seed 7 direct:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed7 \
  --motion-mode none \
  --split-by episode \
  --seed 7
```

Seed 7 future:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_future_seed7 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 7
```

Seed 17 direct:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed17 \
  --motion-mode none \
  --split-by episode \
  --seed 17
```

Seed 17 future:

```bash
python scripts/train_oracle_action_decoder.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl \
  --output-dir outputs/oracle_action_decoder/libero_all_suites_1file_future_seed17 \
  --motion-mode future_delta \
  --split-by episode \
  --seed 17
```

## Results

| seed | branch | train windows | val windows | final val MSE | final val MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | direct context | 6,231 | 1,690 | 0.081479 | 0.160300 |
| 7 | oracle future motion | 6,231 | 1,690 | 0.035064 | 0.085608 |
| 17 | direct context | 6,345 | 1,576 | 0.065460 | 0.149165 |
| 17 | oracle future motion | 6,345 | 1,576 | 0.031770 | 0.084627 |

Relative validation improvement:

| seed | MSE reduction | MAE reduction |
| ---: | ---: | ---: |
| 7 | 56.97% | 46.60% |
| 17 | 51.47% | 43.27% |

## Artifacts

```text
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/summary.json
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/windows.jsonl
outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed7/metrics.json
outputs/oracle_action_decoder/libero_all_suites_1file_future_seed7/metrics.json
outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed17/metrics.json
outputs/oracle_action_decoder/libero_all_suites_1file_future_seed17/metrics.json
```

Companion discussion/report:

```text
docs/agent_qa/2026-06-06-four-suite-test-dataset-usage.md
docs/agent_qa/2026-06-06-four-suite-oracle-action-results.md
```

## Interpretation

This gate is positive. GT future EEF motion is not redundant with direct
context/proprioception for action-chunk decoding on this four-suite small formal
slice.

The result supports the current mechanism route:

```text
context / visual grounding -> future motion prior -> action decoder
```

## Limits

This does not prove visual grounding, cVAE sampling quality, or closed-loop task
success. It only verifies that the oracle future-motion interface has action
value under this slice and MLP decoder.

Flat MSE/MAE should be supplemented with SE(3)-aware translation, rotation, and
gripper metrics before stronger claims.

## Next Decision

Scale the oracle diagnostic to either 2 files per suite or the full four-suite
export, then add SE(3)-aware metrics. Only after the larger-scale oracle gap
remains positive should the project promote learned future-motion priors and
visual grounding experiments.

