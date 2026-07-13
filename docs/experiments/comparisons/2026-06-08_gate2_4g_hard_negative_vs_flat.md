# Gate 2.4g Hard Negative Vs Flat ScoreNet

- Date: 2026-06-08
- Status: completed
- Scope: compare naive hard-negative auxiliary training against the Gate 2.4d
  flat action-MSE ScoreNet baseline.

## Result Table

| branch | action MSE | flat rank | flat top1 | gap closed | promoted? |
| --- | ---: | ---: | ---: | ---: | --- |
| Gate 2.4d flat ScoreNet | 0.040190 | 6.624765 | 0.235384 | 18.36% | yes, current baseline |
| Gate 2.4g hardneg w=0.1 | 0.040344 | 6.943479 | 0.226036 | 14.55% | no |
| Gate 2.4g hardneg w=0.5 | 0.040479 | 7.263220 | 0.210748 | 11.20% | no |

## Decision

The hard-negative idea is not rejected in general, but this minimal
implementation is not enough.

This version used the best `SE(3)+gripper` candidate, excluding the action
oracle, as the hard negative. That negative is structured, but it is not
necessarily tied to real contact, grasp, object progress, or execution
feasibility. It can therefore teach the scorer the wrong contrast.

## Next Mainline

Move to an explicit event/executability proxy:

- detect gripper open/close transitions in the future action chunk;
- detect gripper-state/contact-like changes from proprio when available;
- build candidate penalties for missing or mistimed event transitions;
- combine these proxies with flat action-regret ranking.

This should remain before adding a stronger diffusion/flow action head, because
the current question is still whether GeoMoCo future-motion samples can be
read out into useful control-relevant futures.
