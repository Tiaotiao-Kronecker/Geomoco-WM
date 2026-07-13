# Gate 2.4b Visual cVAE Vs Deterministic Priors

- Date: 2026-06-08
- Status: completed
- Scope: compare first visual-conditioned cVAE prior mean against the
  deterministic visual/action-aware baselines.

## Summary Table

| branch | prior/future MSE | trans L2 | orient L2 | KL | action MSE | action MAE | action trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.3b deterministic single-query lambda 0.030 | 0.000782 | 0.018767 | 0.050640 | n/a | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |
| Gate 2.4a deterministic stepwise lambda 0.030 | 0.000776 | 0.017155 | 0.051082 | n/a | 0.042687 | 0.112300 | 0.014750 | 2.042381 | 0.177896 | 67.53% |
| Gate 2.4b visual cVAE prior mean | 0.000802 | 0.017420 | 0.050785 | 0.000740 | 0.041579 | 0.111667 | 0.014936 | 2.060341 | 0.165615 | 70.74% |

## Readout

The first visual cVAE prior mean is slightly better than the deterministic
single-query default on action MSE:

```text
0.042090 -> 0.041579
```

The direct-to-oracle gap closure improves from `69.26%` to `70.74%`.

The improvement mainly comes from seed 17. Seed 7 is worse than the
deterministic baseline, so this should be treated as a weak positive rather
than a robust breakthrough.

The KL is near zero and posterior/prior metrics are almost identical. This
means the current run does not prove meaningful multimodal latent usage.

## Decision

Keep the deterministic single-query `lambda_action=0.030` branch as the stable
baseline.

Use visual cVAE as the next research branch, but require a calibration gate
before claiming stochastic/multimodal value.

## Next Gate

Gate 2.4c should focus on cVAE calibration:

1. prior sample and best-of-K coverage metrics;
2. KL/free-bits or beta schedule;
3. gripper/contact auxiliary diagnostics;
4. seed stability.
