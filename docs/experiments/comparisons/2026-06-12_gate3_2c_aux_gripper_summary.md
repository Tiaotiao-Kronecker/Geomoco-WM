# Gate 3.2c Auxiliary Gripper Summary

## Question

Can a separate future-gripper auxiliary head repair the open/close transition
bottleneck without blunt transition weighting?

## Result

No. The auxiliary head does not beat the Gate 3.1f baseline.

| branch | readout | overall MSE | transition MSE | transition gripper MSE |
| --- | --- | ---: | ---: | ---: |
| Gate 3.1f baseline | main | 0.034767 | 0.134087 | 0.827336 |
| Gate 3.2c aux w0.3 | main | 0.036377 | 0.134748 | 0.824031 |
| Gate 3.2c aux w0.3 | aux | 0.036388 | 0.133936 | 0.818348 |
| Gate 3.2c aux w1.0 | main | 0.036503 | 0.137979 | 0.850726 |
| Gate 3.2c aux w1.0 | aux | 0.036599 | 0.137685 | 0.848667 |

## Interpretation

The auxiliary gripper branch gives at most a tiny transition gain and worsens
overall action MSE. This suggests the transition bottleneck is not just missing
a separate gripper regressor. It needs explicit event routing or residual
specialization.

## Decision

Do not promote the aux-gripper branch. Next try Gate 3.2d: transition-gated
residual action head or event-routed output layer.
