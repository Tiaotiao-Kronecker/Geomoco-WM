# Oracle Gap And Next-Step Sequencing

- Date: 2026-06-08
- Context: discussion after Gate 2.2b patch-pooled DINOv2 cross-attention
  visual prior.

## Question

Even Gate 2.2b still has a large gap to oracle future motion. How should this
be interpreted? Should the next step be visual controls, or should the project
move directly to multimodal/action-aware methods to close the oracle gap?

## Current Numbers

Mean downstream action metrics:

| branch | action MSE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| direct context | 0.066010 | 0.019024 | 2.233651 | 0.252545 |
| Gate 2.2b patch visual learned prior | 0.049547 | 0.014859 | 2.030450 | 0.222467 |
| oracle future motion | 0.031474 | 0.007466 | 1.048033 | 0.184683 |

Gate 2.2b beats direct context, but still does not match oracle future motion.
It closes `47.67%` of the direct-context to oracle-future-motion action-MSE
gap.

## Interpretation Of The Oracle Gap

The remaining oracle gap is expected and should not be read as failure.

Oracle future motion is privileged information: it gives the action decoder the
ground-truth future EEF trajectory from the demonstration. In contrast, Gate
2.2b predicts that future from current visual/proprio/task context.

Likely reasons for the remaining gap:

1. Oracle future motion is almost a future motion plan, not an ordinary
   observation.
2. Gate 2.2b is deterministic and can average over multiple valid futures.
3. Future EEF motion is not a complete action representation: gripper/contact,
   controller details, and fine orientation adjustments remain under-modeled.
4. Patch pooling is still coarse: 4x4 pooled DINO tokens do not fully localize
   handles, contact surfaces, or grasp points.
5. The training objective is still future-motion MSE, not directly downstream
   action value.

## Decision

The two proposed next steps are sequential, not mutually exclusive.

Immediate next step:

```text
run visual controls
```

Specifically:

```text
shuffled DINO features
agentview-only
eye-in-hand-only
two-camera confirmation
```

Reason: Gate 2.2b is the first strong positive visual result, so the project
must prove the result is not caused by split leakage, task correlation, or
camera-specific shortcut before using it as the cVAE backbone.

After visual controls pass:

```text
use multimodal/action-aware methods to reduce the remaining oracle gap
```

Candidate follow-ups:

```text
action-aware auxiliary loss: motion_loss + lambda * frozen_action_decoder_loss
multimodal cVAE / diffusion / flow future-motion prior
step-wise visual queries for each future step
separate gripper/contact auxiliary branch
```

## Practical Gate Order

Recommended order:

```text
Gate 2.2-control: shuffled/camera controls
Gate 2.3: action-aware deterministic visual prior
Gate 2.4: multimodal visual prior or cVAE smoke
Gate 3: visual-grounded GeoMoCo-cVAE formal run
```

If controls fail, stop and diagnose visual feature leakage/correlation before
adding model complexity.

If controls pass, use Gate 2.2b as the visual grounding route for the next
action-aware or multimodal prior.

