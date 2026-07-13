# Gate 3.4 Controlled Joint Temporal Action-Sequence Decoder

## Purpose

Gate 3.3 showed that a shallow gripper-only additive residual does not improve
the Gate 3.1f/g event-aware predicted-event action head. Gate 3.4 tests a
slightly richer but still controlled alternative:

```text
Keep the Gate 3.1f/g predicted top-4 event/rank/prob sample interface fixed.
Add a small joint temporal action-sequence decoder.
Report both base actions and temporal_actions.
Run attribution controls before promotion.
```

This is not yet a flow or diffusion residual decoder. The goal is to test
whether joint temporal action decoding can consume the same GeoMoCo-WM motion
prior interface better, while preserving attribution.

## Code Changes

Implemented an optional temporal action branch:

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
scripts/audit_predicted_event_mixture_action_head_groups.py
tests/test_motion_prior_action_head.py
```

New model config:

```text
temporal_action_decoder_mode=sequence_mlp
temporal_action_loss_weight=1.0
```

New model output:

```text
temporal_actions [B,H,A]
```

New control config:

```text
future_input_control=real | mean_repeated | context_only
```

`real` preserves existing behavior. `mean_repeated` replaces each K-sample
motion set with the per-window mean repeated K times. `context_only` removes
motion-prior inputs while preserving the same temporal decoder capacity.

## Commands

Main short-budget branch:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
    --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
    --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
    --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
    --output-dir outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed} \
    --event-top-m 4 \
    --num-samples 16 \
    --sample-feature-mode event_rank_prob \
    --temporal-action-decoder-mode sequence_mlp \
    --temporal-action-loss-weight 1.0 \
    --selection-metric temporal_action_mse \
    --epochs 20 \
    --batch-size 64 \
    --seed ${seed} \
    --device cuda \
    --quiet
done
```

Repeated evaluation and group audit were run for the main branch:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/repeated_eval.json \
  --num-eval-passes 5 \
  --device cuda

.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/group_audit.json \
  --num-eval-passes 3 \
  --device cuda
```

Short-budget attribution controls were run with the same decoder:

```text
shuffled_event_rank_prob
rank_prob_only
event_rank_prob + future_input_control=mean_repeated
sample_feature_mode=none + future_input_control=context_only
```

Each control was trained on seeds 7 and 17, then evaluated with 5 repeated
validation passes.

## Results

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | promoted readout | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.4 full aligned | base actions | 0.034303 | 0.149826 | 0.022542 | 0.131697 |
| Gate 3.4 full aligned | temporal_actions | 0.034262 | 0.149383 | 0.022542 | 0.131311 |

The temporal branch is a small positive relative to the same checkpoint's base
output:

```text
overall MSE:    0.034303 -> 0.034262
transition MSE: 0.131697 -> 0.131311
```

It also beats the Gate 3.1f/g reference on overall and transition MSE:

```text
overall MSE:    0.034767 -> 0.034262
transition MSE: 0.134087 -> 0.131311
```

## Controls

Mean over seeds 7 and 17, 5-pass repeated eval, using `temporal_actions`:

| branch | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| full event/rank/prob | 0.034262 | 0.149383 | 0.022542 | 0.131311 |
| shuffled event/rank/prob | 0.035529 | 0.154002 | 0.023499 | 0.135399 |
| rank/prob-only | 0.035875 | 0.156041 | 0.023834 | 0.135893 |
| mean repeated | 0.034414 | 0.149085 | 0.022599 | 0.132199 |
| context-only/no-prior | 0.036642 | 0.156306 | 0.024540 | 0.136922 |

## Interpretation

Gate 3.4 is a controlled small positive. The joint temporal action decoder
improves slightly over its same-checkpoint base output and over the Gate 3.1f/g
reference.

The attribution controls support keeping the aligned event/rank/prob interface:
full aligned beats shuffled event metadata, rank/prob-only, and context-only
no-prior controls.

The important caveat is `mean_repeated`: it is close to full aligned
(`0.034414` vs `0.034262`). This means the current improvement is not strong
evidence that the decoder is using fine-grained sample diversity. The signal is
still more clearly tied to aligned event metadata and the motion-prior mean
structure than to the full K-sample distribution.

## Decision

Promote Gate 3.4 only as a short-budget mechanism-positive result, not as a
new final default.

Keep the interpretation strict:

```text
The controlled joint temporal decoder gives a small gain.
Aligned GeoMoCo-WM event metadata still matters.
Same-capacity no-prior decoding is worse.
Sample diversity attribution remains thin because mean replacement is close.
```

The next mainline should either:

```text
1. strengthen sample-diversity use with set-wise temporal decoding or regret/rank supervision;
2. test a small flow/diffusion residual decoder with the same controls;
3. add richer contact/object/event supervision before increasing decoder capacity further.
```

## Verification

Checks:

```text
.venv/bin/python -m compileall -q scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py
.venv/bin/python -m unittest tests.test_motion_prior_action_head tests.test_predicted_event_mixture_action_head_group_audit tests.test_gripper_boundary_timing_audit
.venv/bin/ruff check scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/audit_predicted_event_mixture_action_head_groups.py tests/test_motion_prior_action_head.py
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_4_temporal_action_shuffled_event_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_4_temporal_action_shuffled_event_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_4_temporal_action_rankprob_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_4_temporal_action_rankprob_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_4_temporal_action_mean_repeated_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_4_temporal_action_mean_repeated_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_4_temporal_action_context_only_seed7/
outputs/motion_prior_action_head/gate3_4_temporal_action_context_only_seed17/
```
