# Gate 1.6 Geodesic Replacement Summary

- Date: 2026-06-06
- Scope: clean replacement for the Gate 1.6 two-file oracle action-decoder
  result after the action-semantics audit.

## Compared Records

| record | role |
| --- | --- |
| `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_se3_metrics.md` | older normalized/split SE(3)-aware table |
| `docs/experiments/runs/2026-06-06_action_semantics_audit_geodesic_metrics.md` | action semantics audit |
| `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_geodesic_replacement.md` | replacement table with physical/geodesic metrics |

## Replacement Result

| seed | MSE reduction | MAE reduction | trans L2 reduction | rot geodesic reduction | gripper MSE reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 50.75% | 44.67% | 60.34% | 53.53% | 28.19% |
| 17 | 53.99% | 47.27% | 61.16% | 52.65% | 25.26% |
| mean | 52.32% | 45.96% | 60.76% | 53.08% | 26.87% |

Mean absolute validation metrics:

| branch | trans L2 (m) | rot geodesic (deg) |
| --- | ---: | ---: |
| direct context | 0.019024 | 2.233651 |
| oracle future motion | 0.007466 | 1.048033 |

## Decision

The oracle future-motion interface remains positive under the corrected
measurement contract. Use this replacement as the clean Gate 1.6 reference for
future learned-prior comparisons.

Next mainline:

```text
Gate 2: learned future-motion prior on the same 2-files-per-suite slice
```
