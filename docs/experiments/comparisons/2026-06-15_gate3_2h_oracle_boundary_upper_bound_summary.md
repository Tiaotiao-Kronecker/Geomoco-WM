# Gate 3.2h Oracle Boundary Upper-Bound Summary

## Question

Can a true boundary mask make transition-local gripper correction good enough
to beat the Gate 3.1f/Gate 3.1g reference?

## Result

Yes as an oracle upper bound, no as a deployable predicted-mask branch.

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.2h base | 0.035145 | 0.154961 | 0.021841 | 0.145678 |
| Gate 3.2h oracle boundary mask | 0.032018 | 0.133072 | 0.021841 | 0.116469 |
| best predicted mask | 0.035201 | 0.155339 | 0.021902 | 0.145723 |

## Interpretation

The oracle mask proves that transition-local gripper correction is the right
mechanism when boundary timing is available. It improves overall MSE, gripper
MSE, and transition MSE without damaging sustain windows.

The predicted mask fails to recover this gain. Low thresholds fire too often;
argmax/high thresholds fire too rarely. Boundary AP remains around `0.098873`
and argmax recall is only `0.012026`.

## Decision

Do not promote Gate 3.2h.

Stop simple sparse boundary CE/residual variants. The next mainline should use
a different temporal localization objective/head or pivot to a richer
temporal/flow action decoder for gripper transitions.
