# Gate 2.4i EEF-Only vs Gripper Interface

- Date: 2026-06-09
- Scope: oracle interface audit for action and close/open transition fidelity.

## Summary

| oracle input | action MSE | gripper MSE | transition acc | step within 1 |
| --- | ---: | ---: | ---: | ---: |
| context only | 0.066010 | 0.252545 | 0.203855 | 0.055590 |
| future gripper only | 0.020666 | 0.000848 | 1.000000 | 1.000000 |
| future EEF only | 0.031474 | 0.184683 | 0.178190 | 0.054159 |
| future EEF + gripper | 0.004202 | 0.000241 | 1.000000 | 1.000000 |

## Core Finding

EEF-only future motion is not enough for gripper/open-close timing.

Adding the oracle future gripper command to future EEF deltas reduces action MSE
by `86.65%` and gripper MSE by `99.87%` relative to EEF-only.

## Mainline Consequence

The next GeoMoCo-WM representation should not be only:

```text
future_delta_ee
```

It should become:

```text
future_delta_ee + future_gripper/event
```

This keeps GeoMoCo focused on world-motion / phase composition, while adding the
missing manipulation-event channel.

