# Gate 3.5b Frozen Post-Hoc Residual Adapter Plan

## Purpose

Gate 3.5a 的 joint residual-flow branch 没有 beat Gate 3.4，而且同 checkpoint
里的 temporal decoder 也退化了。Gate 3.5b 因此把变量拆开：

```text
Freeze promoted Gate 3.4.
Train only a small residual adapter after frozen temporal_actions.
Keep the predicted top-4 event/rank/prob sample interface fixed.
```

The experiment asks whether residual action-sequence modeling is useful when it
cannot perturb the already promoted temporal decoder.

## Minimal Model

For each frozen Gate 3.4 checkpoint:

```text
frozen_features, frozen_temporal_actions = Gate3.4(context, predicted event-mixture samples)
residual = Adapter(frozen_features, frozen_temporal_actions)
adapter_actions = frozen_temporal_actions + residual
```

The adapter is deterministic and small. The initial output should equal
`frozen_temporal_actions` by zero-initializing the final residual layer. No
flow noise is used in this first 3.5b slice.

## First Short-Budget Run

Run only full aligned first:

```text
frozen checkpoints:
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed17/model.pt

epochs: 20
batch_size: 64
adapter_hidden_dims: 256,256
adapter_step_dim: 32
selection_metric: adapter_mse
```

Promotion check before controls:

```text
Gate 3.5b full aligned adapter_mse < Gate 3.4 temporal_action_mse 0.034262
transition MSE should not materially regress from Gate 3.4 transition MSE 0.131311
decoder gain = frozen_temporal_mse - adapter_mse > 0
```

If full aligned fails, archive as negative/neutral and do not spend the full
control budget.

## Attribution Controls

Only if full aligned passes the first check, train matched post-hoc adapters on
the corresponding frozen Gate 3.4 control checkpoints:

```text
full aligned event/rank/prob
shuffled event metadata
rank/prob-only
mean_repeated
context-only/no-prior
```

Then run usage probes:

```text
eval-time mean collapse
permutation sanity
batch mismatch
```

Compute:

```text
decoder gain = frozen temporal_actions MSE - adapter_actions MSE
prior gain = context-only/no-prior adapter MSE - full aligned adapter MSE
metadata gain = metadata-control adapter MSE - full aligned adapter MSE
diversity gain = mean_repeated adapter MSE - full aligned adapter MSE
```

## Interpretation

Positive:

```text
Residual action modeling is useful, but it needs to be decoupled from shared
Gate 3.4 temporal representation learning. Then run controls to verify the
gain still belongs to aligned GeoMoCo-WM samples and event metadata.
```

Negative:

```text
Residual decoder capacity is probably not the bottleneck. Return upstream to
transition/event candidate quality or sample diversity quality before trying a
larger black-box action decoder.
```

