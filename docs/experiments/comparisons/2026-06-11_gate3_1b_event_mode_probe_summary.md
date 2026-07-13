# Gate 3.1b Event Mode Probe Summary

## Question

Is explicit gripper-event timing predictable from aligned visual context?

## Result

Yes.

| variant | macro-F1 | transition F1 | transition timing acc |
| --- | ---: | ---: | ---: |
| task/proprio | 0.306219 | 0.423041 | 0.361899 |
| real visual/proprio | 0.448741 | 0.599939 | 0.458805 |
| shuffled visual/proprio | 0.090006 | 0.191527 | 0.158829 |

## Decision

Proceed to Gate 3.1c event-conditioned cVAE.

## Meaning For Mainline

The project now has evidence for the exact mode axis it wants to expose:

```text
visual context -> gripper/event timing mode -> future_delta_gripper samples
```

This supports the Gate 3.1 story that useful GeoMoCo-WM multimodality should be
mode-structured around event timing, not merely high-variance latent sampling.

## Guardrail

Keep oracle event conditioning separate from deployable predicted-event
mixtures. Oracle-event cVAE is an upper bound, not a deployable model.
