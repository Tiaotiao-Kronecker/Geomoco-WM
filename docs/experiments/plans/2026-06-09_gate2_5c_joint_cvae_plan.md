# Gate 2.5c Joint GeoMoCo-cVAE Plan

- Date: 2026-06-09
- Status: completed
- Position: after Gate 2.5b-joint promoted the deterministic
  `future_delta_ee + future_gripper/event` output space.

## Purpose

Upgrade the stochastic GeoMoCo-cVAE output from EEF-only:

```text
future_delta_ee
```

to the joint manipulation-event target:

```text
future_delta_ee + future_gripper/event
```

The deterministic baseline is Gate 2.5b-joint:

```text
action MSE: 0.040688
transition accuracy: 0.560270
```

## Required Checks

1. Prior mean action value:
   compare cVAE prior mean to deterministic joint baseline.

2. Sample coverage:
   compare random samples and oracle best-of-K over K=16.

3. Visual attribution:
   compare real DINO patchpool with shuffled DINO patchpool.

4. Event fidelity:
   evaluate prior-mean gripper transition metrics.

## Initial Config

Follow Gate 2.4c stochastic settings:

```text
latent_dim = 32
beta_kl = 0.001
beta_kl_start = 0.0
beta_kl_warmup_epochs = 5
free_bits = 0.02
epochs = 20
batch_size = 64
```

Use the Gate 2.5b-joint action-aware setting:

```text
motion_mode = future_delta_gripper
action_aware_loss_weight = 0.300
```

Test `prior_recon_weight`:

```text
1.0
0.5
```

## Decision Rule

If prior mean beats the deterministic baseline, promote cVAE prior mean as the
next deployable branch.

If prior mean does not beat deterministic but best-of-K is strong and real
visual beats shuffled visual, promote the cVAE sample space but move next to a
joint sample readout/scorer gate.

