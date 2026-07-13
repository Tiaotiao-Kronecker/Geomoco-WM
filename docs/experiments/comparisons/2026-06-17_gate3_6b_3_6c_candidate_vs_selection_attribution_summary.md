# Gate 3.6b/3.6c Candidate-vs-Selection Attribution Summary

## Question

Was Gate 3.6a's transition improvement caused by upstream transition-reserve
candidate allocation, or by the different checkpoint selection metric?

## Result

The transition improvement is real, but the candidate-allocation attribution is
not supported.

Mean over seeds 7 and 17:

| branch | candidate policy | selection metric | temporal MSE | transition MSE | sustain MSE |
| --- | --- | --- | ---: | ---: | ---: |
| Gate 3.4 | top-k | temporal_action_mse | 0.034262 | 0.131311 | 0.022542 |
| Gate 3.6a | transition_reserve t=0.15 | temporal_action_transition_mse | 0.036391 | 0.126421 | 0.025539 |
| Gate 3.6b | transition_reserve t=0.25 | temporal_action_mse | 0.034262 | 0.131311 | 0.022542 |
| Gate 3.6b | transition_reserve t=0.35 | temporal_action_mse | 0.034262 | 0.131311 | 0.022542 |
| Gate 3.6c | top-k | temporal_action_transition_mse | 0.036391 | 0.126421 | 0.025539 |

## Trigger Diagnostic

The current `transition_reserve` rule only acts when top-4 has no transition
candidate. That condition never occurs in the current train/validation split:

| seed | split | total | true transition windows | top-4 without transition | reserve triggers |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | train | 12974 | 1296 | 0 | 0 |
| 7 | val | 3544 | 403 | 0 | 0 |
| 17 | train | 13305 | 1374 | 0 | 0 |
| 17 | val | 3213 | 325 | 0 | 0 |

## Attribution Correction

Gate 3.6c exactly matches Gate 3.6a with `event_candidate_policy=topk`. So the
old 3.6a sentence:

```text
transition_reserve candidate allocation improves transition MSE
```

should be read as:

```text
transition-MSE checkpoint selection improves transition MSE, with an
overall/sustain trade-off.
```

The matched 3.6a controls still describe the selected checkpoint's dependence
on prior/metadata/diversity, but they do not prove that the `transition_reserve`
policy caused the gain.

## Decision

Archive Gate 3.6b/3.6c as an attribution correction:

```text
Gate 3.6b: fixed-threshold reserve at top-4 is inert.
Gate 3.6c: transition-MSE checkpoint selection recreates the 3.6a trade-off.
Promotion: no.
```

Next mainline should avoid more threshold tuning unless the candidate policy
actually changes selected candidates. If the goal remains transition repair,
move to either explicit transition/sustain Pareto checkpoint analysis or a
candidate policy based on transition timing diversity / downstream action
regret.
