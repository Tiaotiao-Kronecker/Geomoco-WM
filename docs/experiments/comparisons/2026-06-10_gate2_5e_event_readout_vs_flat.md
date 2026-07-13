# Gate 2.5e Event Readout vs Flat ScoreNet

- Date: 2026-06-10
- Scope: seed-7 pilot over joint `future_delta_gripper` cVAE samples

## Mean Readout Direction

| method | action MSE | event acc | transition acc | step@1 | event error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 2.5d flat | 0.045230 | 0.893065 | 0.622739 | 0.240310 | 1.841783 |
| event `w=0.05` | 0.045455 | 0.893357 | 0.602067 | 0.250646 | 1.821387 |
| event `w=0.10` | 0.045710 | 0.898019 | 0.612403 | 0.271318 | 1.764277 |
| event HN `w=0.05, hn=0.10` | 0.047043 | 0.904429 | 0.645995 | 0.284238 | 1.714452 |

## Takeaway

The event-aware objective is doing something real, but not yet what the main
policy interface needs.

As event alignment gets stronger, event metrics improve, but action MSE gets
worse. The hard-negative variant improves event alignment the most and hurts
action MSE the most. This is a clean sign of target conflict, not merely
undertraining.

## Mainline Decision

Do not expand this pilot into a full two-seed/shuffled matrix unless a later
architecture change makes action and event ranking compatible.

The immediate next step should be one of:

- stronger temporal/action-regret readout over candidate action chunks;
- cVAE/sample-space training that jointly preserves EEF geometry and gripper
  timing;
- closed-loop or rollout-level executability diagnostics where event timing is
  evaluated as an effect, not only as a scalar readout target.

