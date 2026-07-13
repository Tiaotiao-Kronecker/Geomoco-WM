# Gate 2.2 Visual Controls Summary

- Date: 2026-06-08
- Scope: shuffled visual features and camera ablations for Gate 2.2b patch
  cross-attention.

## Mean Metrics

| branch | future-motion MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.1 suite/task prior | 0.000929 | 0.072501 | 0.155573 | 0.020841 | 2.106491 | 0.279382 |
| Shuffled two-camera patch | 0.000985 | 0.075521 | 0.154999 | 0.020578 | 2.127742 | 0.300477 |
| Agentview-only patch | 0.000804 | 0.054732 | 0.127885 | 0.016336 | 2.014087 | 0.233136 |
| Eye-in-hand-only patch | 0.000799 | 0.050853 | 0.124119 | 0.015433 | 2.024622 | 0.224925 |
| Two-camera patch reference | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| Direct context | n/a | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| Oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Direct-To-Oracle Gap Closure

Action-MSE gap closure is computed as:

```text
(direct_context_mse - branch_mse) / (direct_context_mse - oracle_future_motion_mse)
```

| branch | action MSE | closure |
| --- | ---: | ---: |
| Shuffled two-camera patch | 0.075521 | -27.54% |
| Gate 2.1 suite/task prior | 0.072501 | -18.79% |
| Agentview-only patch | 0.054732 | 32.66% |
| DINO global visual prior | 0.053628 | 35.85% |
| Eye-in-hand-only patch | 0.050853 | 43.89% |
| Two-camera patch reference | 0.049547 | 47.67% |

## Pass Criteria Readout

The visual controls pass.

The shuffled branch is much worse than aligned two-camera visual grounding:

```text
0.075521 vs 0.049547 action MSE
```

The shuffled branch also performs worse than direct context:

```text
0.075521 > 0.066010 action MSE
```

This is evidence that correct visual alignment, not just feature dimension or
model capacity, drives the Gate 2.2 gain.

Both real single-camera branches are useful:

```text
agentview-only action MSE: 0.054732
eye-in-hand-only action MSE: 0.050853
direct-context action MSE: 0.066010
```

Eye-in-hand is slightly stronger than agentview on this slice, and two-camera
fusion is still the best branch. The current best route remains two-camera
patch cross-attention.

## Interpretation

This result upgrades Gate 2.2 from a positive visual result to a reliable
mechanism result:

```text
aligned visual grounding > shuffled visual grounding
aligned visual grounding > direct context
two-camera patch grounding >= eye-in-hand-only > agentview-only > global DINO
```

The remaining gap to oracle future motion is still large, but now it is a
meaningful modeling target rather than an attribution problem. The current
predictor sees useful visual evidence but still predicts a single deterministic
future-motion chunk with MSE supervision. That limits multimodality,
contact/gripper coupling, and action-relevant trajectory selection.

## Next Mainline

Do not jump straight to GeoMoCo-cVAE as a paper claim only because visual
controls passed. The next step should first attack the remaining oracle gap:

```text
aligned two-camera DINO patch tokens
  + action-aware auxiliary loss or frozen action-decoder loss
  + optional multimodal prior
  -> then connect the validated visual path into GeoMoCo-cVAE
```

Concrete next experiments:

1. Add an action-aware auxiliary term to the future-motion prior using the
   frozen Gate 1.6 action decoder.
2. Add step-wise or multi-query visual attention so future steps can attend to
   different visual evidence.
3. Add a separate gripper/contact branch because gripper improvements lag the
   geometric action improvements.
4. After these diagnostics, promote the visual branch into GeoMoCo-cVAE and
   compare deterministic, cVAE, and action-aware variants.
