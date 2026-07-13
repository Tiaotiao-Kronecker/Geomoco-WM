# Gate 3.0c Usage Audit Summary

## Question

Does the Gate 3.0 action head actually use multimodal future-motion samples, or
does it collapse the sample set to a mean feature?

## Main Result

Mean action MSE across seeds:

| source | original K=16 | mean repeated | subset K=4 | batch mismatch |
| --- | ---: | ---: | ---: | ---: |
| real visual cVAE | 0.037061 | 0.041893 | 0.040585 | 0.317347 |
| shuffled visual cVAE | 0.042665 | 0.053407 | 0.048454 | 0.221706 |

The original set beats mean replacement and K=4 subset. Batch-mismatching the
sample set destroys performance. Permuting sample order leaves the output
unchanged, as expected for a set model.

## Interpretation

The action head is using sample-set information. It is not only reading a mean
future.

The important nuance:

```text
shuffled samples are more diverse, but less useful.
real samples are less diverse, but better aligned and more action-useful.
```

So the target is not generic diversity. The target is aligned multimodal
future-motion structure, especially around gripper/event timing.

## Mainline Decision

Gate 3.0c supports moving from generic readout tuning to structured multimodal
prior design:

```text
Gate 3.1: mode-structured / event-aware future-motion prior
```

The next branch should make event/transition modes explicit or at least measure
them as first-class diagnostics, while preserving:

```text
real-vs-shuffled controls
sample-set usage audit
downstream action-head evaluation
```
