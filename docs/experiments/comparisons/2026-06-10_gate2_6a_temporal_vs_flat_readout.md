# Gate 2.6a Temporal Readout vs Gate 2.5d Flat ScoreNet

- Date: 2026-06-10
- Scope: real visual joint cVAE, seeds 7 and 17

## Action Readout

| method | prior action MSE | scorer action MSE | oracle action MSE | selected rank |
| --- | ---: | ---: | ---: | ---: |
| Gate 2.5d flat ScoreNet | 0.043816 | 0.043414 | 0.022192 | 5.637193 |
| Gate 2.6a temporal ScoreNet | 0.043816 | 0.043636 | 0.022158 | 5.551251 |

## Event Diagnostics

| method | event acc | transition acc | step@1 |
| --- | ---: | ---: | ---: |
| Gate 2.5d flat ScoreNet | 0.900561 | 0.647706 | 0.264299 |
| Gate 2.6a temporal ScoreNet | 0.899412 | 0.636113 | 0.289056 |

## Takeaway

Temporal scoring is not enough by itself.

The temporal scorer improves selected oracle rank and transition step@1, but it
does not improve the main downstream action metric. Since action MSE regresses
from `0.043414` to `0.043636`, this version should not be promoted.

## Next Decision

The next readout iteration should not only change encoder capacity. It should
change the target/interface more directly:

```text
score = calibrated prediction of downstream action regret
```

or use set-wise candidate comparison so the scorer can reason about relative
candidate quality instead of scoring each candidate independently.

