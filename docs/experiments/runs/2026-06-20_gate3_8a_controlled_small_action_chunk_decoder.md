# Gate 3.8a Controlled Small Action-Chunk Decoder

## Purpose

Gate 3.8a is the short-budget follow-up after the Gate 3.7a soft event-time
latent negative. It tests whether a slightly stronger but still controlled
action-chunk decoder can model close/open transitions as part of the full
action sequence.

The branch keeps the GeoMoCo-WM prior/sample interface fixed:

```text
predicted top-4 event mixture
K = 16 samples
sample_feature_mode = event_rank_prob
selection_metric = temporal_action_mse
no event-time latent
no flow residual
no candidate policy change
```

## Code Change

Added a new `MotionPriorActionHead` temporal decoder mode:

```text
temporal_action_decoder_mode = temporal_transformer
```

It reuses the existing `temporal_actions [B,H,A]` output and temporal-action
loss. Compared with Gate 3.4 `sequence_mlp`, it lets step tokens interact
through a small TransformerEncoder before per-step action prediction.

Touched files:

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
tests/test_motion_prior_action_head.py
```

The eval and post-hoc loader paths continue to load the mode through saved
`model_config`.

## Config

Seed-7 full aligned run:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_8a_temporal_transformer_top4_k16_seed7 \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --temporal-action-decoder-mode temporal_transformer \
  --temporal-action-loss-weight 1.0 \
  --selection-metric temporal_action_mse \
  --epochs 20 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --quiet
```

Repeated evaluation:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_8a_temporal_transformer_top4_k16_seed7/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_8a_temporal_transformer_top4_k16_seed7/repeated_eval.json \
  --num-eval-passes 5 \
  --device cuda
```

## Stop Rule

Continue only if seed 7 beats the Gate 3.4 seed-7 full-aligned reference:

```text
temporal_action_mse < 0.036502
transition_mse      < 0.137827
sustain_mse         <= about 0.0236
```

## Result

Seed 7, 5-pass repeated validation:

| branch | temporal MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.4 seed 7 sequence_mlp | 0.036502 | 0.137827 | 0.023501 | 0.164389 |
| Gate 3.7a seed 7 soft event-time | 0.037941 | 0.147882 | 0.023835 | 0.171543 |
| Gate 3.8a seed 7 temporal_transformer | 0.038275 | 0.162345 | 0.022356 | 0.176774 |

Gate 3.8a fails the stop rule:

```text
temporal_action_mse is worse by      +0.001773
transition MSE is worse by           +0.024518
gripper MSE is worse by              +0.012385
sustain MSE is better by             -0.001145
```

The temporal transformer does slightly improve sustain MSE and keeps SE(3)
metrics close to Gate 3.4, but it badly worsens transition and gripper errors.

## Interpretation

This is a clean short-budget negative for the minimal temporal-transformer
action-chunk readout. Adding step-token self-attention on top of the existing
global feature does not solve the close/open transition bottleneck. In this
configuration it appears to trade toward easier sustain/continuous-motion
regions while damaging sparse gripper transition windows.

This does not rule out action-chunk policies in general. It rules out the
cheapest controlled version:

```text
same features + same loss + tiny temporal transformer readout
```

The result is consistent with earlier gates: transition errors are not fixed by
small readout-capacity changes unless the model gets a reliable transition
timing/contact signal or a stronger policy objective that models action
multimodality more directly.

## Decision

Stop Gate 3.8a at seed 7. Do not run seed 17 or attribution controls for this
branch.

Artifact:

```text
outputs/motion_prior_action_head/gate3_8a_temporal_transformer_top4_k16_seed7/
```

Next useful choices are:

```text
1. fix event-time supervision first, then reintroduce it only if positive
   close/open timing improves before action training;
2. move to a stronger but still controlled action-chunk policy objective, such
   as small conditional flow/diffusion over the full action chunk, with the same
   decoder/prior/metadata/diversity controls;
3. improve upstream transition/contact candidate quality using a policy that
   actually changes transition timing diversity, not fixed top-4 reserve.
```

## Verification

Checks run:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/train_predicted_event_mixture_posthoc_residual_adapter.py tests/test_motion_prior_action_head.py
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py ... --temporal-action-decoder-mode temporal_transformer --dry-run
```
