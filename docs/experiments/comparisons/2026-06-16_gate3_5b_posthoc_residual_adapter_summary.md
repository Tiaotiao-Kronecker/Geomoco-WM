# Gate 3.5b Post-Hoc Residual Adapter Summary

## Question

If Gate 3.5a failed because joint residual-flow training damaged the shared
temporal decoder, does a frozen Gate 3.4 post-hoc residual adapter recover a
clean residual-action gain?

## Result

Partly yes, but not enough to promote.

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | adapter MSE | transition MSE | interpretation |
| --- | ---: | ---: | --- |
| Gate 3.4 temporal reference | 0.034262 | 0.131311 | previous promoted reference |
| Gate 3.5b full aligned | 0.033374 | 0.133339 | better overall, worse transition |
| shuffled event control | 0.034274 | 0.135600 | full aligned beats it |
| rank/prob-only control | 0.034099 | 0.135325 | full aligned beats it |
| mean repeated control | 0.033604 | 0.136726 | close overall, worse transition |
| context-only/no-prior control | 0.033728 | 0.131042 | close overall, best transition |

## Gain Ledger

```text
decoder gain  = frozen temporal - full adapter = +0.000762
prior gain    = context-only - full aligned    = +0.000354
metadata gain = shuffled - full aligned        = +0.000900
metadata gain = rank/prob-only - full aligned  = +0.000726
diversity gain = mean_repeated - full aligned  = +0.000230
```

The adapter improves overall MSE and full aligned remains best among trained
controls, but the attribution gains are thin. In particular, context-only
nearly catches up.

## Transition Caveat

The main bottleneck was transition/gripper timing. Gate 3.5b does not solve it:

```text
Gate 3.4 temporal transition MSE: 0.131311
Gate 3.5b full transition MSE:    0.133339
context-only transition MSE:      0.131042
```

So the overall gain likely comes from easier sustain/SE(3)/translation repair,
not from fixing close/open transition behavior.

## Usage Audit

Within the same trained full-aligned adapters, eval-time sample usage is real:

| eval-time variant | adapter MSE |
| --- | ---: |
| original | 0.033277 |
| mean repeated | 0.043023 |
| permuted samples | 0.033277 |
| subset K=4 | 0.070474 |
| batch mismatch | 0.314030 |

This mirrors Gate 3.4b: the checkpoint uses K-sample structure at runtime, but
matched training controls can still adapt around reduced diversity.

## Decision

Do not promote Gate 3.5b as the new default. Archive it as:

```text
overall-positive
post-hoc residual mechanism-positive
attribution-thin
transition-negative
```

The next mainline should not simply scale adapter capacity. It should constrain
the residual to the transition failure mode or improve upstream event/transition
candidate quality, while preserving the same decoder/prior/metadata/diversity
controls.

