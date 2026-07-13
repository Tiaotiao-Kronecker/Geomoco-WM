# Gate 3.7a Soft Event-Time Latent Short-Budget Run

## Purpose

Gate 3.7a tests the cleaner version of option A from the event-time vs
action-chunk discussion:

```text
predict a soft close/open event-time distribution
condition the existing sequence_mlp temporal action decoder on that distribution
avoid hard argmax boundary routing
```

This is a small diagnostic branch, not a full new policy. The expansion rule was
deliberately strict: first test full aligned only, then run matched attribution
controls only if the branch preserves overall/sustain quality while improving
transition timing.

## Config

Seed-7 full aligned run:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_7a_soft_event_time_top4_k16_seed7 \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --temporal-action-decoder-mode sequence_mlp \
  --temporal-action-loss-weight 1.0 \
  --event-time-conditioning-mode soft_boundary \
  --event-time-loss-weight 0.1 \
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
  --checkpoint outputs/motion_prior_action_head/gate3_7a_soft_event_time_top4_k16_seed7/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_7a_soft_event_time_top4_k16_seed7/repeated_eval.json \
  --num-eval-passes 5 \
  --device cuda
```

## Result

Seed 7, 5-pass repeated validation:

| branch | temporal MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.4 seed 7 full aligned | 0.036502 | 0.137827 | 0.023501 | 0.164389 |
| Gate 3.7a seed 7 full aligned | 0.037941 | 0.147882 | 0.023835 | 0.171543 |

Gate 3.7a seed 7 also misses the pre-declared expansion criteria:

```text
required temporal_action_mse <= 0.0344
actual   temporal_action_mse  = 0.037941

required temporal_action_transition_mse < 0.131311
actual   temporal_action_transition_mse  = 0.147882

required temporal_action_sustain_mse <= 0.0230
actual   temporal_action_sustain_mse  = 0.023835
```

Event-time diagnostics:

| metric | value |
| --- | ---: |
| event_time_ce | 0.279181 |
| event_time_accuracy | 0.937472 |
| event_time_event_fraction | 0.057562 |
| event_time_close_accuracy | 0.119656 |
| event_time_open_accuracy | 0.016919 |
| event_time_close_within1 | 0.350704 |
| event_time_open_within1 | 0.156866 |
| event_time_entropy | 0.136281 |

## Interpretation

This is a short-budget negative for this specific minimal A implementation.
The branch does not merely fail to improve transition timing; it worsens
overall temporal action MSE, transition MSE, sustain MSE, and gripper MSE
relative to the Gate 3.4 seed-7 full-aligned reference.

The event-time auxiliary metrics explain part of the failure. Overall
`event_time_accuracy` is high, but positive close/open timing is weak because
event labels are sparse:

```text
event_time_event_fraction = 0.057562
close within-1            = 0.350704
open within-1             = 0.156866
```

So the soft event-time token is probably dominated by no-event confidence and
does not provide a reliable transition-time signal to the decoder. Injecting a
low-quality soft boundary distribution into the temporal decoder can therefore
hurt the same `sequence_mlp` branch that Gate 3.4 already trained cleanly.

This does not prove that every event-time latent design is wrong. It does say
that the current unbalanced close/open/no-event CE plus direct soft-token
conditioning is not worth expanding.

## Decision

Stop Gate 3.7a at seed 7 under the user's short-budget instruction.

Do not run seed 17 or matched controls for this branch because the full-aligned
branch failed the expansion gate. Preserve the artifact only as a diagnostic:

```text
outputs/motion_prior_action_head/gate3_7a_soft_event_time_top4_k16_seed7/
```

If returning to option A later, the next variant should first fix the event-time
objective itself, for example with positive-event-balanced/focal supervision and
separate no-event calibration. Otherwise the cleaner next mainline is to move
toward a small motion-prior-conditioned action-chunk policy while keeping the
same decoder/prior/metadata/diversity controls.
