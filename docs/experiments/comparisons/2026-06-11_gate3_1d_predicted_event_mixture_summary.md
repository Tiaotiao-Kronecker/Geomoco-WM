# Gate 3.1d Predicted Event Mixture Summary

## Question

Can a visual event-mode predictor replace oracle event labels for the
event-conditioned joint cVAE?

## Result

Partially yes.

| branch | prior action MSE | best-of-K action MSE | sample pair L2 |
| --- | ---: | ---: | ---: |
| unconditional joint cVAE | 0.043816 | 0.022139 | 0.612986 |
| oracle-event cVAE | 0.018448 | 0.014656 | 0.179034 |
| predicted top-1 | 0.050072 | 0.042023 | 0.265841 |
| predicted top-2 | 0.045272 | 0.027537 | 1.510387 |
| predicted top-4 | 0.042992 | 0.015228 | 2.704827 |

## Interpretation

Predicted top-4 almost recovers oracle-event best-of-K coverage:

```text
oracle-event best-of-K action MSE: 0.014656
predicted top-4 best-of-K action MSE: 0.015228
```

This is strong evidence that the visually predicted event-mode mixture contains
action-useful futures. It is also evidence that the sample set is wide and noisy:
the sample pair L2 grows to 2.704827, and the prior/readout action MSE remains
far above the oracle-event prior.

## Decision

Keep the mainline story as:

```text
GeoMoCo-WM proposes a structured set of future_delta_gripper hypotheses.
The downstream action head/planner must learn to consume that set.
```

Do not promote predicted top-4 as a standalone policy interface yet. Promote it
as the next proposal source for Gate 3 action-head evaluation.
