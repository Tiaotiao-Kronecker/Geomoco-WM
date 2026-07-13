# Visual Grounding Design And Gate 2.2 Plan

- Date: 2026-06-07
- Context: discussion before returning from Gate 2.1 to the visual-grounded
  GeoMoCo-WM mainline.

## Discussion Summary

The downstream action value metric is no longer only flat action MSE. Formal
action-decoder gates now report:

```text
flat action MSE / MAE
translation_m_l2
rotation_geodesic_deg
gripper_mse
```

The action rotation metric is SE(3)-aware: LIBERO normalized rotation actions
are scaled by the audited `0.5rad` controller scale, mapped through SO(3), and
compared with geodesic distance. Translation actions are scaled by the audited
`0.05m` controller scale.

Future EEF-motion orientation metrics are still coordinate metrics, not SO(3)
geodesic metrics, because the exported observation orientation representation
is separate from the audited controller action rotvec representation.

## Patch Tokens

For DINO/ViT visual features:

```text
patch_tokens: [T, N, D]
```

means:

- `T`: number of history/context frames.
- `N`: number of spatial image patches per frame.
- `D`: visual feature dimension per patch token.

For 224x224 RGB with a ViT patch size of 14, `N = 16 * 16 = 256`. With local
`dinov2_vits14_reg`, `D = 384`.

Two-camera features can be stored separately:

```text
agentview_patch_tokens: [T, N, D]
eye_in_hand_patch_tokens: [T, N, D]
```

or concatenated into:

```text
visual_tokens: [T, 2N, D]
```

## Grounding Computation

The intended visual grounding path is:

```text
DINO(image history) -> visual tokens
proprio + task -> query
query attends visual tokens -> g_t
[proprio, task, g_t] -> predicted future_delta_ee
```

Interpretation:

1. DINO converts RGB history into visual tokens.
2. Proprioception and task metadata form a query that asks which image regions
   matter under the current robot state and task.
3. The query attends over visual tokens and produces a grounded visual summary
   `g_t`.
4. The future-motion prior predicts future EEF delta from proprioception, task,
   and `g_t`.

Even though `g_t` is produced by a proprio/task-conditioned query, proprio/task
should still be passed to the final predictor. `g_t` is a visual summary, not a
complete robot state. Passing proprio/task again is a skip connection that
preserves EEF pose, gripper state, joint state, and task identity.

## One-Hot Metadata

One-hot represents categorical labels without imposing a fake numerical order.

Example with four suites:

```text
libero_goal -> [0, 0, 1, 0]
```

Gate 2.1 used `suite_task` one-hot with dimension 8 because the
2-files-per-suite slice has 8 suite/task combinations.

## Execution Order

Mainline order:

```text
Gate 2.2a: DINO global-token visual prior
Gate 2.2b: DINO patch-token cross-attention grounding
then: visual-grounded GeoMoCo-cVAE
```

Gate 2.2a should keep attribution simple:

```text
frozen DINOv2 global token + proprio + suite_task -> future EEF motion
```

Gate 2.2b should add patch-level grounding:

```text
proprio/task query attends DINO patch tokens -> g_t -> future EEF motion
```

Do not enter cVAE until a visual-grounded deterministic prior is evaluated
against the direct-context and oracle-future-motion bounds.

## GPU Note

Local elevated execution can see the RTX 5090:

```text
torch 2.12.0+cu130
cuda_available True
device_count 1
device_name NVIDIA GeForce RTX 5090
```

The default restricted execution context hides CUDA, so formal Gate 2.2 runs
should use approved GPU-visible execution and `--device cuda`.

