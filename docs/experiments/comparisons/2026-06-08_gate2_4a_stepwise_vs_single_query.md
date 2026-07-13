# Gate 2.4a Stepwise Vs Single-Query Visual Grounding

- Date: 2026-06-08
- Status: completed
- Scope: decide whether stepwise multi-query visual attention should replace
  the Gate 2.3b single-query action-aware default.

## Comparison

| branch | future MSE | future trans L2 | future orient L2 | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.3a single-query lambda 0.010 | 0.000770 | 0.016763 | 0.049942 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 | 66.12% |
| Gate 2.3b single-query lambda 0.030 | 0.000782 | 0.018767 | 0.050640 | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |
| Gate 2.4a stepwise-query lambda 0.030 | 0.000776 | 0.017155 | 0.051082 | 0.042687 | 0.112300 | 0.014750 | 2.042381 | 0.177896 | 67.53% |

## Readout

Stepwise visual attention does what it was partly meant to do: it improves
future-motion translation L2 compared with the stronger single-query
`lambda_action=0.030` branch.

It does not improve the main downstream action metric. The action MSE is worse
than the current default by `0.000597`, or roughly `1.42%` relative.

## Decision

Keep the default deterministic action-value prior as:

```text
visual_fusion = cross_attention
lambda_action = 0.030
```

Treat stepwise multi-query attention as an optional geometry-balanced variant,
not the default.

## Implication For Mainline

This result argues against spending another long deterministic ablation on
query structure alone. The next mainline should move to stochastic /
multimodal future-motion modeling, where the missing oracle gap is more likely
to come from ambiguity and contact/gripper phase than from only the number of
visual attention queries.
