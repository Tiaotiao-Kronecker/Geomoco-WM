# Oracle Action Gate Scale-Up Comparison

- Date: 2026-06-06
- Scope: compare the one-file and two-file four-suite oracle action-decoder
  gates.

## Compared Runs

| gate | record | dataset | status |
| --- | --- | --- | --- |
| Gate 1.5 | `docs/experiments/runs/2026-06-06_gate1_5_four_suite_oracle_action_decoder.md` | 1 file per suite, 7,921 windows | positive |
| Gate 1.6 | `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_se3_metrics.md` | 2 files per suite, 16,518 windows | positive |

## Overall Validation Improvement

| gate | seed | MSE reduction | MAE reduction |
| --- | ---: | ---: | ---: |
| Gate 1.5 | 7 | 56.97% | 46.60% |
| Gate 1.5 | 17 | 51.47% | 43.27% |
| Gate 1.6 | 7 | 50.75% | 44.67% |
| Gate 1.6 | 17 | 53.99% | 47.27% |

## Decision

The oracle future-motion interface remains useful after scaling the data slice.
The direct-context vs oracle-future gap is not a one-task or one-file artifact.

Promote the next mainline step:

```text
learned future-motion prior -> same action-decoder gate
```

Do not jump directly to final visual-grounded claims. The next learned model
should be judged against both:

```text
direct context lower bound
oracle future-motion upper/interface bound
```

## Extra Signal From Gate 1.6

SE(3)-aware metrics show that oracle future motion mostly helps geometric action
recovery:

```text
translation MSE reduction: 82.56% to 83.93%
rotation MSE reduction: 75.08% to 77.25%
gripper MSE reduction: 25.26% to 28.19%
```

This suggests a split modeling plan:

```text
future EEF motion prior: primary geometric branch
gripper/contact prediction: separate auxiliary branch or supervision
```

