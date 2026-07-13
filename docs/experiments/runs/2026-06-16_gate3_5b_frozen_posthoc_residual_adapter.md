# Gate 3.5b Frozen Post-Hoc Residual Adapter

- Date: 2026-06-16
- Status: completed, mixed-positive but not promoted as the next default
- Gate: 3.5b
- Purpose: isolate residual action-sequence modeling from Gate 3.5a's shared
  joint-training interference by freezing Gate 3.4 and training only a small
  residual adapter.

## Motivation

Gate 3.5a failed because the flow branch did not improve `temporal_actions`,
and its same-checkpoint temporal branch also regressed from Gate 3.4. Gate 3.5b
therefore freezes the promoted Gate 3.4 checkpoints:

```text
frozen_features, frozen_temporal_actions = Gate3.4(context, samples)
adapter_actions = frozen_temporal_actions + Adapter(frozen_features, frozen_temporal_actions)
```

The adapter starts from exact frozen-temporal behavior through zero-init of the
final residual layer.

## Config

```text
script: scripts/train_predicted_event_mixture_posthoc_residual_adapter.py
eval: scripts/evaluate_predicted_event_mixture_posthoc_residual_adapter.py
usage audit: scripts/audit_predicted_event_mixture_posthoc_residual_adapter_usage.py
frozen checkpoints: Gate 3.4 full and matched controls
adapter hidden dims: 256,256
adapter step dim: 32
epochs: 20
batch size: 64
lr: 3e-4
seeds: 7, 17
device: cuda
repeated eval passes: 5
usage audit passes: 3
```

## Commands

Full aligned:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/train_predicted_event_mixture_posthoc_residual_adapter.py \
    --frozen-action-head-checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/model.pt \
    --output-dir outputs/motion_prior_action_head/gate3_5b_posthoc_residual_top4_k16_seed${seed} \
    --epochs 20 \
    --batch-size 64 \
    --adapter-hidden-dims 256,256 \
    --adapter-step-dim 32 \
    --device cuda \
    --quiet

  .venv/bin/python scripts/evaluate_predicted_event_mixture_posthoc_residual_adapter.py \
    --checkpoint outputs/motion_prior_action_head/gate3_5b_posthoc_residual_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_5b_posthoc_residual_top4_k16_seed${seed}/repeated_eval.json \
    --num-eval-passes 5 \
    --batch-size 64 \
    --device cuda
done
```

Matched controls use the same command with frozen checkpoints:

```text
gate3_4_temporal_action_shuffled_event_top4_k16_seed*
gate3_4_temporal_action_rankprob_top4_k16_seed*
gate3_4_temporal_action_mean_repeated_top4_k16_seed*
gate3_4_temporal_action_context_only_seed*
```

Usage audit:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/audit_predicted_event_mixture_posthoc_residual_adapter_usage.py \
    --checkpoint outputs/motion_prior_action_head/gate3_5b_posthoc_residual_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_5b_posthoc_residual_top4_k16_seed${seed}/usage_audit.json \
    --num-eval-passes 3 \
    --batch-size 64 \
    --device cuda
done
```

## Results

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | adapter MSE | frozen temporal MSE | decoder gain | gripper MSE | transition MSE | sustain MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full aligned | 0.033374 | 0.034135 | +0.000762 | 0.148705 | 0.133339 | 0.021289 |
| shuffled event | 0.034274 | 0.035451 | +0.001177 | 0.152785 | 0.135600 | 0.022055 |
| rank/prob-only | 0.034099 | 0.035682 | +0.001583 | 0.152767 | 0.135325 | 0.021902 |
| mean repeated | 0.033604 | 0.034253 | +0.000649 | 0.149761 | 0.136726 | 0.021155 |
| context-only/no-prior | 0.033728 | 0.036603 | +0.002876 | 0.144737 | 0.131042 | 0.021969 |

Attribution ledger:

```text
decoder gain  = frozen temporal - full adapter = +0.000762
prior gain    = context-only - full aligned    = +0.000354
metadata gain = shuffled - full aligned        = +0.000900
metadata gain = rank/prob-only - full aligned  = +0.000726
diversity gain = mean_repeated - full aligned  = +0.000230
```

Against the previous Gate 3.4 baseline:

```text
Gate 3.4 temporal MSE:             0.034262
Gate 3.5b full adapter MSE:        0.033374
overall delta:                    +0.000888

Gate 3.4 temporal transition MSE:  0.131311
Gate 3.5b full transition MSE:     0.133339
transition delta:                 -0.002028
```

## Usage Audit

Mean over seeds 7 and 17, 3-pass eval-time audit on full-aligned adapters:

| eval-time variant | adapter MSE |
| --- | ---: |
| original | 0.033277 |
| mean repeated | 0.043023 |
| permuted samples | 0.033277 |
| subset K=4 | 0.070474 |
| batch mismatch | 0.314030 |

The adapter is permutation invariant and strongly sensitive to eval-time mean
collapse and batch mismatch. Runtime usage of aligned sample distributions is
real. The caveat is the matched trained `mean_repeated` control remains close
to full aligned.

## Interpretation

Gate 3.5b is a useful mixed result:

```text
1. Post-hoc residual action modeling is useful for overall MSE.
2. Freezing Gate 3.4 prevents the 3.5a shared-representation regression.
3. The full aligned branch remains best among matched trained controls.
4. However, the context-only adapter nearly catches up, so prior attribution is thin.
5. The transition/gripper bottleneck is not solved; full aligned transition MSE
   regresses from Gate 3.4 and is worse than context-only in this run.
```

This supports the adapter mechanism but does not justify promoting it as the
next default transition-solving decoder.

## Limits

This is not evidence that a larger flow/diffusion policy should be promoted.
The adapter improves mostly sustain/SE(3)/translation behavior and can learn a
large amount without motion-prior samples. The key GeoMoCo-WM claim remains
weaker than in Gate 3.4 because `prior gain` and `diversity gain` are small
under matched training controls.

## Next Decision

Do not scale this exact post-hoc adapter as the main transition branch. The
next clean step should make the residual adapter transition-aware without
letting it become a generic context-only BC repair, for example by adding a
constrained transition-local objective or gating residual application by a
deployable event/transition confidence signal, with the same attribution
controls.

