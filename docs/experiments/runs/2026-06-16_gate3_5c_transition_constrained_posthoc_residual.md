# Gate 3.5c Transition-Constrained Post-Hoc Residual

- Date: 2026-06-16
- Status: completed, transition-negative; do not expand full controls
- Gate: 3.5c
- Purpose: test whether the Gate 3.5b post-hoc residual adapter can be made
  less generic by applying residual corrections only where deployable
  event-probe confidence says a close/open transition is likely.

## Motivation

Gate 3.5b showed that frozen post-hoc residual adaptation improves overall MSE,
but the gain does not target the main bottleneck:

```text
Gate 3.5b full adapter MSE:        0.033374
Gate 3.4 temporal MSE:             0.034262
Gate 3.5b full transition MSE:     0.133339
Gate 3.4 temporal transition MSE:  0.131311
context-only adapter MSE:          0.033728
```

Gate 3.5c keeps the frozen Gate 3.4 checkpoint and residual adapter setup, but
adds a deployable transition-probability gate:

```text
raw_residual = Adapter(frozen_features, frozen_temporal_actions)
gate = leak + (1 - leak) * predicted_transition_prob
adapter_actions = frozen_temporal_actions + gate * raw_residual
```

`predicted_transition_prob` is computed from the event-probe top-M probability
mass assigned to cVAE event classes whose labels are transition events.

## Config

```text
script: scripts/train_predicted_event_mixture_posthoc_residual_adapter.py
eval: scripts/evaluate_predicted_event_mixture_posthoc_residual_adapter.py
frozen checkpoints: Gate 3.4 full aligned checkpoints
event_top_m: 4
num_samples: 16
sample_feature_mode: event_rank_prob
future_input_control: real
adapter hidden dims: 256,256
adapter step dim: 32
epochs: 20
batch size: 64
lr: 3e-4
selection metric: adapter_transition_mse
residual_gate_mode: predicted_transition_prob
residual_gate_threshold: none
residual_leak: 0.05
seeds: 7, 17
device: cuda
repeated eval passes: 5
```

The `none` gate mode preserves Gate 3.5b behavior. The `oracle_transition` gate
mode exists only as a diagnostic upper bound and was not needed for promotion.

## Commands

Full aligned:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/train_predicted_event_mixture_posthoc_residual_adapter.py \
    --frozen-action-head-checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/model.pt \
    --output-dir outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed${seed} \
    --epochs 20 \
    --batch-size 64 \
    --adapter-hidden-dims 256,256 \
    --adapter-step-dim 32 \
    --selection-metric adapter_transition_mse \
    --residual-gate-mode predicted_transition_prob \
    --residual-leak 0.05 \
    --device cuda \
    --quiet

  .venv/bin/python scripts/evaluate_predicted_event_mixture_posthoc_residual_adapter.py \
    --checkpoint outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed${seed}/repeated_eval.json \
    --num-eval-passes 5 \
    --batch-size 64 \
    --device cuda
done
```

## Promotion Check

The plan required full aligned to pass before spending the full matched control
budget:

```text
adapter_transition_mse <= Gate 3.4 temporal transition MSE 0.131311
adapter_mse <= Gate 3.4 temporal MSE 0.034262
decoder_gain_transition_mse >= 0
```

## Results

Per seed, 5-pass repeated eval:

| seed | adapter MSE | frozen temporal MSE | decoder gain | adapter transition MSE | frozen transition MSE | transition gain | gate mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.035843 | 0.036310 | +0.000467 | 0.138685 | 0.136764 | -0.001921 | 0.280880 |
| 17 | 0.031778 | 0.031961 | +0.000183 | 0.127932 | 0.125068 | -0.002864 | 0.267381 |

Mean over seeds 7 and 17:

| metric | value |
| --- | ---: |
| adapter MSE | 0.033810 |
| frozen temporal MSE | 0.034135 |
| decoder gain MSE | +0.000325 |
| adapter transition MSE | 0.133308 |
| frozen temporal transition MSE | 0.130916 |
| decoder gain transition MSE | -0.002392 |
| adapter sustain MSE | 0.021803 |
| adapter gripper MSE | 0.149784 |
| residual gate mean | 0.274131 |
| residual gate min | 0.050028 |
| residual gate max | 0.993391 |

Against the Gate 3.4 promoted reference:

```text
Gate 3.4 temporal MSE:             0.034262
Gate 3.5c adapter MSE:             0.033810
overall delta:                    +0.000452

Gate 3.4 temporal transition MSE:  0.131311
Gate 3.5c adapter transition MSE:  0.133308
transition delta:                 -0.001997
```

Against Gate 3.5b:

```text
Gate 3.5b adapter MSE:             0.033374
Gate 3.5c adapter MSE:             0.033810
Gate 3.5b transition MSE:          0.133339
Gate 3.5c transition MSE:          0.133308
```

The predicted transition gate slightly reduces the generic adapter's strength
and gives a tiny transition improvement over 3.5b, but it remains worse than
Gate 3.4 and worse than the frozen temporal branch inside the same checkpoints.

## Attribution Controls

The full control matrix was intentionally not expanded because the full-aligned
promotion check failed on transition. The controls remain mandatory for any
future positive decoder:

```text
decoder gain = frozen temporal/base - richer decoder
prior gain = context-only/no-prior - full aligned
metadata gain = shuffled/rank-prob-only - full aligned
diversity gain = mean_repeated - full aligned
```

Skipping full controls here preserves budget and avoids over-interpreting a
branch whose main target metric is already negative.

## Interpretation

Gate 3.5c is a useful negative:

```text
1. Transition-probability gating preserves a small overall adapter gain.
2. It does not repair transition/gripper timing.
3. The residual family still improves easier sustain/geometry behavior more
   than the transition bottleneck.
4. The failure is not just "adapter too unconstrained"; deployable transition
   confidence is not yet strong enough to make the residual action model
   transition-useful.
```

The clean next mainline is upstream event/transition candidate quality:
improve the candidate proposal or event-confidence source before trying another
larger action residual decoder.

## Limits

This run does not rule out an oracle transition gate or a substantially
different residual family. It does show that the deployable predicted-event
gate available in the current Gate 3.1f/g interface is insufficient as a
transition-local repair signal.

## Artifacts

```text
outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed7/model.pt
outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed7/metrics.json
outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed7/repeated_eval.json

outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed17/model.pt
outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed17/metrics.json
outputs/motion_prior_action_head/gate3_5c_predgate_l005_transition_top4_k16_seed17/repeated_eval.json
```

## Next Decision

Do not promote Gate 3.5c and do not spend full controls on it. Move to upstream
event/transition candidate quality while preserving the same attribution
ledger for any future positive decoder:

```text
decoder / prior / metadata / diversity controls stay fixed.
```
