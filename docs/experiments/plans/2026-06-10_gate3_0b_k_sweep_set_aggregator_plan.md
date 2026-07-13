# Gate 3.0b K Sweep And Set Aggregator Ablation Plan

## Purpose

Gate 3.0a showed that a downstream action head can use real visual cVAE
`future_delta_gripper` sample sets better than context-only, prior mean, and
shuffled sample controls. The gain over prior mean is still modest. Gate 3.0b
tests whether that gain is limited by the number of sampled futures or by the
set aggregation architecture.

## Fixed Contract

- Dataset: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl`
- Motion mode: `future_delta_gripper`
- Split: episode-level
- Seeds: `7`, `17`
- Frozen priors: Gate 2.5c real/shuffled visual joint cVAE checkpoints
- Action head does not receive raw DINO features
- Stochastic sample-set metrics use repeated validation eval

## K Sweep

Use `context_attention`, the Gate 3.0a default aggregator.

| K | real cVAE samples | shuffled cVAE samples |
| ---: | --- | --- |
| 4 | run | run |
| 8 | run | run |
| 16 | reuse Gate 3.0a | reuse Gate 3.0a |
| 32 | run | run |

Promotion signal:

```text
real improves with K or stays stable
shuffled remains worse than real
best K beats Gate 3.0a K=16 mean MSE 0.036675
```

## Aggregator Ablation

Fix `K=16`.

| aggregator | meaning |
| --- | --- |
| `mean_pool` | average sample tokens, no context query |
| `context_attention` | Gate 3.0a default; context/task query attends samples |
| `multi_query_attention` | multiple context-conditioned queries attend samples, pooled before action decode |

Promotion signal:

```text
best aggregator beats context_attention K=16
best aggregator keeps real > shuffled control
```

## Interpretation Rules

- If larger K improves real but not shuffled, sample diversity is useful.
- If larger K does not help, the current action head may already saturate or the
  samples may not contain additional action-useful variation.
- If `mean_pool` matches attention, the head may not be using set structure.
- If `multi_query_attention` improves, future work can justify richer planning
  heads without jumping directly to flow/diffusion action policies.

## Stop Criteria

Do not add raw visual features or a stronger diffusion/flow action head in this
gate. Those are later baselines. Gate 3.0b is only about whether the existing
motion-prior sample set is being read out effectively.
