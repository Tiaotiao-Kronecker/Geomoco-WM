# Multi-Query Visual Predictor Role And Mainline Position

- Date: 2026-06-08
- Context: after Gate 2.4a stepwise / multi-query visual predictor.

## Question

What is the role of the multi-query visual predictor in the mainline?

## Answer

The multi-query visual predictor is a mechanism probe and architecture
diagnostic, not the final objective.

It asks whether the current visual grounding bottleneck is caused by using only
one visual query for the whole future horizon.

Previous single-query route:

```text
proprio + task -> one query
query attends DINO patch tokens -> one grounded token g_t
[proprio, task, g_t] -> full future_delta_ee chunk
```

Multi-query route:

```text
proprio + task -> base query
base query + step embedding[k] -> query_k
query_k attends DINO patch tokens -> grounded token g_{t,k}
[proprio, task, g_{t,k}, step embedding[k]] -> future_delta_ee[k]
```

The purpose is to test whether different future steps should attend to
different visual evidence.

## Gate 2.4a Result

```text
single-query lambda 0.030 action MSE: 0.042090
multi-query  lambda 0.030 action MSE: 0.042687
single-query lambda 0.030 future trans L2: 0.018767
multi-query  lambda 0.030 future trans L2: 0.017155
```

The multi-query branch improves future-motion translation geometry but does
not improve the main downstream action-value metric.

## Mainline Decision

Do not promote multi-query as the default deterministic branch.

Current default remains:

```text
visual_fusion = cross_attention
lambda_action = 0.030
```

Keep multi-query as:

- a geometry-balanced variant;
- a possible cVAE / stochastic-prior ingredient;
- evidence that query granularity alone is not the main remaining bottleneck.

## Next Mainline

Move to stochastic / multimodal future-motion priors instead of more
deterministic query-only ablations. The next branch should test whether
multi-modality can reduce the remaining oracle gap that deterministic future
prediction cannot close.
