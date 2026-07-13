# Four-Suite Oracle Action-Decoder Results

- Date: 2026-06-06
- Project: `Geomoco-WM`
- Gate: four-suite small formal oracle action-decoder diagnostic

## Purpose

This run tests whether future EEF motion contains actionable information beyond
direct proprioceptive/context state.

The comparison is deliberately attribution-clean:

```text
direct context baseline:
  context -> action chunk

GT future motion oracle:
  context + future EEF delta -> action chunk
```

No DINO, cVAE, ZipMotion, AMPLIFY, diffusion policy, or flow-matching decoder is
used in this gate. The goal is to verify whether the motion interface is worth
learning before training larger visual-grounded future-motion models.

## Dataset Slice

Source root:

```text
/home/user/dataset/libero_official
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

Export summary:

| item | value |
| --- | ---: |
| suites | 4 |
| HDF5 files | 4 |
| episodes / demos | 200 |
| windows | 7,921 |
| frames | 33,201 |
| dropped short episodes | 0 |
| warnings | 0 |
| `windows.jsonl` size | 33M |
| `episodes.jsonl` size | 184K |

Per-suite windows:

| suite | task file | episodes | windows |
| --- | --- | ---: | ---: |
| `libero_spatial` | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 50 | 1,169 |
| `libero_object` | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 50 | 1,857 |
| `libero_goal` | `open_the_middle_drawer_of_the_cabinet` | 50 | 1,663 |
| `libero_10` | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 50 | 3,232 |

Dataset tensors:

| branch | context dim | motion dim | action dim | horizon |
| --- | ---: | ---: | ---: | ---: |
| direct context | 15 | 0 | 7 | 8 |
| GT future motion | 15 | 48 | 7 | 8 |

## Training Setup

Common settings:

```text
model: MLP ActionDecoder
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
split: episode-level
seeds: 7, 17
```

Episode-level split keeps all windows from one demonstration entirely in either
train or validation, reducing adjacent-window leakage.

The runs used CPU in the default restricted execution context.
`torch.cuda.is_available()` returned `False` there with `torch 2.10.0+cu128`,
`cuda_version=12.8`, and `device_count=0`.

The machine and Python environment are CUDA-capable when run with normal GPU
access. An elevated check saw `NVIDIA GeForce RTX 5090`, driver `580.95.05`,
system CUDA `13.0`, and the same Python environment reported
`torch.cuda.is_available() == True`, `cuda_version=12.8`, `device_count=1`.

This is not a blocker for this small MLP gate, but heavier DINO, cVAE, or
full-suite training should run from a GPU-visible shell or approved execution
mode.

## Results

| seed | branch | train windows | val windows | final val MSE | final val MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | direct context | 6,231 | 1,690 | 0.081479 | 0.160300 |
| 7 | GT future motion | 6,231 | 1,690 | 0.035064 | 0.085608 |
| 17 | direct context | 6,345 | 1,576 | 0.065460 | 0.149165 |
| 17 | GT future motion | 6,345 | 1,576 | 0.031770 | 0.084627 |

Relative validation improvement:

| seed | MSE reduction | MAE reduction |
| ---: | ---: | ---: |
| 7 | 56.97% | 46.60% |
| 17 | 51.47% | 43.27% |

Artifacts:

```text
outputs/libero_windows/libero_all_suites_1file_all_demos_h8/
outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed7/
outputs/oracle_action_decoder/libero_all_suites_1file_future_seed7/
outputs/oracle_action_decoder/libero_all_suites_1file_direct_seed17/
outputs/oracle_action_decoder/libero_all_suites_1file_future_seed17/
```

## Interpretation

This gate is positive. Across two episode-level seeds, GT future EEF deltas
substantially improve action-chunk decoding over direct context alone.

The result supports the current GeoMoCo-WM direction:

```text
visual/proprio/task context -> future motion prior -> action decoder
```

It also justifies continuing toward learned future-motion models, because the
oracle motion representation is not redundant with the local proprioceptive
context in this four-suite slice.

## Limits

This does not prove closed-loop task success, visual grounding, DINO feature
quality, cVAE sampling quality, or policy robustness.

It also does not prove that an MLP is the final action decoder. The MLP is only
the first clean diagnostic; stronger temporal, diffusion-policy, flow-matching,
or MeanFlow-style decoders should be introduced after the representation and
CUDA environment are ready.

## Next Decisions

1. Fix or select the CUDA-enabled Python environment before heavy training.
2. Add SE(3)-aware action/motion metrics alongside flat MSE/MAE.
3. Scale the oracle gate to either 2 files per suite or the full four-suite
   export, depending on JSONL size and training-loop timing.
4. If the oracle gap remains positive at the larger scale, start the first
   learned future-motion model before DINO-heavy visual grounding.
