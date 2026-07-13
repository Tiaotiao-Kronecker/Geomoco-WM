# Gate 2.4a Stepwise Multi-Query Visual Predictor Plan

- Date: 2026-06-08
- Status: planned
- Gate: Gate 2.4a
- Purpose: test whether horizon-step-specific visual queries improve the
  action-executable future-motion prior before moving to stochastic cVAE /
  multimodal priors.

## Motivation

Gate 2.2b and Gate 2.3 used one proprio/task query:

```text
proprio + suite_task -> query
query attends DINO patch tokens -> one grounded token g_t
[proprio, suite_task, g_t] -> full future_delta_ee chunk
```

This gives the whole horizon a single visual summary. That is clean but may be
too weak for multi-step manipulation because early, middle, and late future
motion can depend on different visual evidence.

Gate 2.4a changes only the visual fusion shape:

```text
proprio + suite_task -> base query
base query + step embedding[k] -> query_k
query_k attends DINO patch tokens -> grounded token g_{t,k}
[proprio, suite_task, g_{t,k}, step embedding[k]] -> future_delta_ee[k]
```

The model still predicts deterministic future EEF deltas and still uses the
frozen action-decoder auxiliary loss. This makes the comparison isolated:
single-query cross-attention vs stepwise multi-query cross-attention.

## Dataset And Baselines

Dataset:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Visual cache:

```text
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

Reference rows:

| branch | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.2b single-query MSE-only | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 | 47.67% |
| Gate 2.3a single-query lambda 0.010 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 | 66.12% |
| Gate 2.3b single-query lambda 0.030 | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |

## Experiment Matrix

Primary run:

| setting | value |
| --- | --- |
| visual fusion | `stepwise_cross_attention` |
| action-aware lambda | `0.030` |
| seed(s) | `7`, `17` |
| epochs | `20` |
| batch size | `64` |
| hidden dims | `256,256` |
| split | episode |

Optional reference if results are ambiguous:

| setting | value |
| --- | --- |
| visual fusion | `stepwise_cross_attention` |
| action-aware lambda | `0.010` |
| seed(s) | `7`, `17` |

## Pass / Stop Criteria

Pass if the primary run improves mean action MSE over Gate 2.3b
`lambda_action=0.030` without a large future-motion geometry regression.

Guardrails:

- future-motion MSE should stay close to `~0.00078`;
- future translation L2 should not degrade far beyond the existing `0.018767`
  unless action-space gains are clear and stable;
- action translation L2, SO(3) geodesic rotation, and gripper MSE should be
  reported separately.

Stop or redesign if:

- action MSE worsens relative to the single-query lambda `0.030` baseline;
- action MSE improves only through a large future-motion geometry distortion;
- one seed improves and the other collapses.

## Next Step If Positive

Promote stepwise visual grounding into the first stochastic / multimodal prior:

```text
visual stepwise queries -> deterministic mean branch
visual stepwise queries -> cVAE prior/posterior condition
action-aware loss remains an auxiliary value signal
```

If the stepwise deterministic branch is neutral, move directly to multimodal
cVAE while keeping single-query `lambda_action=0.030` as the baseline.
