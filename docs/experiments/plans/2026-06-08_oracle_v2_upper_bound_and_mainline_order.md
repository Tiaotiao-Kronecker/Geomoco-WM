# Oracle v2 Upper-Bound Plan And Mainline Order

- Date: 2026-06-08
- Status: planned
- Role: side-track upper-bound calibration, not the immediate mainline

## Why Oracle Future Motion Still Has Headroom

The current oracle branch is:

```text
context + GT future EEF delta -> frozen ActionDecoder -> action chunk
```

It is an upper-bound diagnostic for the future-motion interface, but it is not
a full action oracle. GT future EEF motion does not uniquely determine the
OSC_POSE action chunk because inverse dynamics is ambiguous and because the
current motion representation omits contact, gripper phase, object state, and
visual change.

Current mean reference:

| branch | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct context | 0.066010 | 0.147124 | 0.019024 | 2.233651 | 0.252545 |
| Gate 2.2b aligned visual learned prior | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| Current oracle future EEF motion | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Oracle v2 Ladder

| Stage | Added information | Purpose |
| --- | --- | --- |
| Oracle v1 | GT future EEF delta | Current clean motion-interface upper bound. |
| Oracle v2a | GT future EEF delta + GT future gripper/contact proxy | Test whether gripper/contact omission explains residual action error. |
| Oracle v2b | GT future EEF delta + future visual or object-change proxy | Test whether scene/object dynamics are needed for inverse dynamics. |
| Oracle v2c | Stronger temporal inverse dynamics decoder | Test whether the MLP action decoder is the bottleneck. |
| Oracle v2d | Step-wise inverse dynamics: state/action at each transition | Test whether chunk-level decoding hides transition-level ambiguity. |

## Mainline Position

Oracle v2 is not the next immediate mainline step. It should be used as an
upper-bound calibration track when either of these conditions holds:

- learned visual prior approaches the current oracle future-motion bound;
- downstream improvements stall and it is unclear whether the bottleneck is
  the learned prior or the oracle/action-decoder interface itself.

The current mainline order is:

```text
Gate 2.2c visual controls
  -> Gate 2.3 action-aware visual future-motion prior
  -> Gate 2.4 multimodal / stochastic future-motion prior
  -> Oracle v2 upper-bound calibration if needed
  -> GeoMoCo-cVAE with validated visual/action-aware route
```

## Immediate Next Step

Add an action-aware auxiliary loss to the aligned two-camera DINO patch
cross-attention future-motion prior:

```text
motion_loss = MSE(pred_future_ee_delta, gt_future_ee_delta)
action_loss = MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)
total_loss = motion_loss + lambda_action * action_loss
```

This tests whether the learned prior can predict future motion that is not only
coordinate-accurate, but also useful to the downstream action interface.

## Pass / Stop Criteria

Promote Gate 2.3 if it improves downstream action metrics over Gate 2.2b
without a severe future-motion collapse:

```text
Gate 2.2b action MSE: 0.049547
target: lower action MSE, stable translation/rotation metrics
```

If action-aware training improves action MSE but worsens physical future-motion
metrics sharply, keep it as an auxiliary diagnostic rather than the default
future-motion prior.
