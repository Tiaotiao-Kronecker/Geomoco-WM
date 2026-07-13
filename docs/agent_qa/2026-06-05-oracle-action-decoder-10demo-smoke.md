# Oracle Future-Motion Action Decoder 10-Demo Smoke

- Date: 2026-06-05
- Dataset: `outputs/libero_windows/libero_goal_task0_10demo/windows.jsonl`
- Task: `open_the_middle_drawer_of_the_cabinet`
- Scope: 10 demos, 307 windows, context length 2, future/action horizon 16,
  stride 4

## Purpose

This smoke checks the first executable-interface question:

```text
Does GT future EEF motion help action prediction over direct context?
```

This is not yet a publishable result. It is a small, single-task, single-seed,
offline diagnostic.

## Compared Conditions

Both runs use:

```text
ActionDecoder MLP
hidden dims: 128,128
epochs: 30
batch size: 64
device: cpu
seed: 7
train ratio: 0.8
```

Conditions:

```text
direct context:
  anchor EEF + gripper + joint -> action chunk

oracle future motion:
  anchor EEF + gripper + joint + GT future EEF delta -> action chunk
```

## Results

| Condition | Val MSE | Val MAE |
| --- | ---: | ---: |
| direct context | `0.035896` | `0.113958` |
| GT future EEF delta | `0.017163` | `0.088143` |

Relative improvement:

```text
Val MSE reduction: 52.19%
Val MAE reduction: 22.65%
```

## Interpretation

This is a positive early signal for the executable-interface gate:

```text
GT future motion -> action
```

The current result says the future EEF delta interface can carry useful action
information beyond anchor EEF, gripper, and joint context on the drawer smoke
subset.

## Limits

- Only one task file.
- Only 10 demos.
- Only one random split and one seed.
- No visual context yet.
- No stronger action decoder yet.
- The future EEF motion is ground truth, so this does not prove a learned
  GeoMoCo-WM prior.

## Next Step

Run the same comparison on more demos/tasks and add a stronger temporal decoder
only after the MLP diagnostic stays positive.
