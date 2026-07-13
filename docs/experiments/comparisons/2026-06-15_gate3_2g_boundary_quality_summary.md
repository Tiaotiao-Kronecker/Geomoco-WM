# Gate 3.2g Boundary Quality Summary

## Question

Can boundary-quality auditing and a transition-local residual repair the Gate
3.2f boundary-start branch?

## Result

Not enough.

| branch | readout | overall MSE | gripper MSE | sustain MSE | transition MSE | boundary AP | argmax recall | argmax precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 3.1f reference | default | 0.034767 | 0.150052 | 0.022793 | 0.134087 | - | - | - |
| Gate 3.2f all-classes | step | 0.035605 | 0.156003 | 0.023363 | 0.137172 | 0.096503 | 0.007233 | 0.254444 |
| Gate 3.2g positive-only | step | 0.035204 | 0.155630 | 0.022986 | 0.136703 | 0.090235 | 0.001634 | 0.015176 |
| Gate 3.2g posw20 | step | 0.035778 | 0.154816 | 0.022957 | 0.142132 | 0.095948 | 0.532779 | 0.094722 |

## Interpretation

`positive_only` residual blending slightly improves transition MSE and
transition gripper MSE, but it still hurts overall metrics and does not beat
the Gate 3.1f/Gate 3.1g reference.

Positive CE weighting solves the wrong part of the problem. It raises boundary
recall, but precision stays low and action quality degrades. In short:

```text
the model can be made to fire on boundary steps,
but the fires are not localized or precise enough to drive action residuals.
```

## Decision

Do not promote Gate 3.2g.

Stop adding simple sparse CE variants for this deterministic residual head.
The next mainline should either:

```text
1. use oracle boundary masks to train a direct transition-local gripper
   trajectory correction, then test whether predicted masks can recover it; or
2. move to a richer temporal action head / flow-style decoder after documenting
   that simple deterministic routing is exhausted.
```
