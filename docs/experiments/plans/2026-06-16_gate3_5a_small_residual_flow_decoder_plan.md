# Gate 3.5a Small Residual Flow Decoder Plan

## Purpose

Gate 3.4 showed a small controlled gain from a joint temporal action decoder.
Gate 3.4b showed the full checkpoint uses K-sample diversity at runtime, but
trained `mean_repeated` remains close. Gate 3.4c showed that motion-regret
sample scoring does not convert this diversity usage into better action value.

Gate 3.5a therefore changes the action trajectory model itself, while keeping
the attribution contract intact:

```text
Add a small residual flow decoder after temporal_actions.
Do not replace the base action head or the Gate 3.4 temporal decoder.
Keep the predicted top-4 event/rank/prob sample interface fixed.
```

## Minimal Decoder

Use a rectified-flow style residual branch:

```text
base output:      actions
temporal output:  temporal_actions
flow output:      flow_actions = temporal_actions + predicted_residual
```

Training target:

```text
residual_target = ground_truth_actions - temporal_actions
z ~ N(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) * z + t * residual_target
velocity_target = residual_target - z
```

The model predicts `velocity`. For the deployable one-step deterministic readout
used in validation:

```text
predicted_residual = flow_velocity(x_t=0, t=0, cond)
flow_actions = temporal_actions + predicted_residual
```

This is intentionally smaller than a full diffusion policy. It is a residual
adapter over the existing Gate 3.4 action sequence.

## Controls Kept Fixed

Use the same predicted-event mixture interface:

```text
event_top_m=4
num_samples=16
sample_feature_mode=event_rank_prob
temporal_action_decoder_mode=sequence_mlp
temporal_action_loss_weight=1.0
sample_score_mode=none
```

No motion-regret scorer in this branch.

## First Short-Budget Run

Run only full aligned first:

```text
seeds: 7, 17
epochs: 20
batch_size: 64
selection_metric: flow_action_mse
flow_action_decoder_mode: rectified_mlp
flow_action_loss_weight: 1.0
```

Promotion check before controls:

```text
Gate 3.5a full aligned flow_action_mse < Gate 3.4 temporal_action_mse 0.034262
Gate 3.5a transition MSE should improve or at least not regress materially
relative to Gate 3.4 transition MSE 0.131311.
```

If full aligned fails this check, stop and archive as negative/neutral.

## Attribution Ledger

Only if full aligned passes the first check, run:

```text
full aligned event/rank/prob
shuffled event metadata
rank/prob-only
trained mean_repeated
context-only/no-prior
same-checkpoint eval-time mean collapse
permutation sanity
batch mismatch
```

Compute:

```text
decoder gain = same-checkpoint temporal_actions MSE - flow_actions MSE
prior gain = context-only/no-prior flow MSE - full aligned flow MSE
metadata gain = metadata-control flow MSE - full aligned flow MSE
diversity gain = mean_repeated flow MSE - full aligned flow MSE
```

Interpretation rule:

```text
If flow improves full aligned but context-only improves equally, it is decoder
capacity gain rather than GeoMoCo-WM gain.

If full aligned uniquely beats context-only, shuffled/rank-prob controls, and
mean_repeated, the richer decoder preserves attribution.
```

## Stop Conditions

Stop without full controls if:

```text
flow_action_mse >= 0.034262 on the two-seed full-aligned short run
or transition MSE regresses enough to erase the Gate 3.4 transition gain.
```

Do not add another boundary localizer or motion-regret scorer in this slice.
