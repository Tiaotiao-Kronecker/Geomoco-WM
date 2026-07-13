# Gate 3.1c Oracle Event cVAE Summary

## Question

Does explicit event-mode conditioning make the joint cVAE sample space more
action-useful?

## Result

Yes, as an upper bound.

| branch | prior action MSE | best-of-K action MSE | prior gripper MSE |
| --- | ---: | ---: | ---: |
| unconditional joint cVAE | 0.043816 | 0.022139 | 0.183879 |
| shuffled-event cVAE | 0.042406 | 0.021296 | 0.169936 |
| oracle-event cVAE | 0.018448 | 0.014656 | 0.030296 |

## Decision

Proceed to predicted event-mode mixture sampling.

## Mainline Meaning

The useful multimodal axis is indeed gripper/event timing. When the cVAE is told
the correct event mode, it produces much more action-useful
`future_delta_gripper` candidates. Shuffled event labels do not reproduce the
gain, so this is not just extra capacity.

## Guardrail

Oracle-event conditioning is privileged. Do not present it as the deployed
GeoMoCo-WM method. It is a diagnostic upper bound that motivates:

```text
visual event-mode predictor -> top-M event modes -> event-conditioned cVAE samples
```
