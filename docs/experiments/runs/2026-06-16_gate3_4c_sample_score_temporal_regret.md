# Gate 3.4c Sample-Score Temporal Regret Supervision

## Purpose

Gate 3.4 was a small positive for joint temporal action decoding, and Gate 3.4b
showed that the full-aligned checkpoint uses K-sample diversity at evaluation
time. The remaining caveat was stricter:

```text
The separately trained mean_repeated control stayed close to full aligned.
```

Gate 3.4c tests a minimal way to make K-sample diversity more causally useful
under the matched training setup:

```text
Add explicit set-wise candidate-comparison supervision.
Keep the predicted top-4 event/rank/prob interface fixed.
Do not move yet to a larger flow/diffusion residual decoder.
```

## Design

Implemented an optional sample scorer inside `MotionPriorActionHead`:

```text
sample_score_mode=action_regret
```

When enabled:

```text
1. each candidate future-motion sample token receives a scalar score;
2. softmax(sample_scores) aggregates the K sample tokens;
3. an auxiliary candidate-comparison loss supervises those scores.
```

The first target was deliberately cheap and prior-facing:

```text
sample_score_target=motion_regret
regret_k = mean_square(sample_motion_k - oracle_future_motion)
target_probs = softmax(-regret / temperature)
loss = CE(target_probs, scorer_logits)
```

This asks the action head to prefer samples closer to the oracle future-motion
target without using action labels to define the score target.

## Code

Changed:

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
tests/test_motion_prior_action_head.py
```

New config fields:

```text
sample_score_mode=none|action_regret
sample_score_loss_weight
sample_score_target=motion_regret|temporal_action_regret
sample_score_loss_type=soft_ce
sample_score_temperature
```

New diagnostics:

```text
sample_score_loss
sample_score_top1_accuracy
sample_score_expected_regret
sample_score_expected_vs_best_gap
sample_score_entropy
```

## Commands

Full-aligned short-budget runs:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
    --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
    --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
    --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
    --output-dir outputs/motion_prior_action_head/gate3_4c_sample_score_top4_k16_seed${seed} \
    --event-top-m 4 \
    --num-samples 16 \
    --sample-feature-mode event_rank_prob \
    --temporal-action-decoder-mode sequence_mlp \
    --temporal-action-loss-weight 1.0 \
    --sample-score-mode action_regret \
    --sample-score-loss-weight 0.1 \
    --sample-score-target motion_regret \
    --sample-score-loss-type soft_ce \
    --sample-score-temperature 0.05 \
    --selection-metric temporal_action_mse \
    --epochs 20 \
    --batch-size 64 \
    --seed ${seed} \
    --device cuda \
    --quiet
done
```

Repeated evaluation:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
    --checkpoint outputs/motion_prior_action_head/gate3_4c_sample_score_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_4c_sample_score_top4_k16_seed${seed}/repeated_eval.json \
    --num-eval-passes 5 \
    --device cuda
done
```

## Results

5-pass repeated eval:

| seed | base MSE | temporal MSE | gripper MSE | transition MSE | sustain MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.036934 | 0.036560 | 0.166132 | 0.142806 | 0.022929 |
| 17 | 0.033432 | 0.032998 | 0.142273 | 0.140433 | 0.020908 |
| mean | 0.035183 | 0.034779 | 0.154203 | 0.141620 | 0.021918 |

Scorer diagnostics, 5-pass repeated eval:

| seed | score loss | top-1 acc | expected regret | gap to best | entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 2.285905 | 0.201975 | 0.057100 | 0.053229 | 2.233603 |
| 17 | 2.256417 | 0.252537 | 0.050407 | 0.047304 | 2.232798 |
| mean | 2.271161 | 0.227256 | 0.053754 | 0.050267 | 2.233200 |

## Comparison To Gate 3.4

Mean over seeds 7 and 17:

| branch | temporal MSE | transition MSE | interpretation |
| --- | ---: | ---: | --- |
| Gate 3.4 full aligned | 0.034262 | 0.131311 | previous controlled temporal decoder |
| Gate 3.4 mean_repeated control | 0.034414 | 0.132199 | close trained mean-only control |
| Gate 3.4 context-only/no-prior | 0.036642 | 0.136922 | same-capacity no-prior control |
| Gate 3.4c sample-score full aligned | 0.034779 | 0.141620 | explicit motion-regret scoring |

Gate 3.4c does not beat Gate 3.4. It also worsens transition MSE relative to
the Gate 3.4 full-aligned branch and the Gate 3.4 mean-repeated control.

## Interpretation

This is an informative negative/neutral short-budget result.

The sample-score target does learn a non-random candidate signal, but the signal
does not translate into better action-sequence prediction. The likely reason is
that `motion_regret` is only an approximate proxy for downstream action value:
a sample can be closer to oracle future motion on average while still not
providing the best gripper transition or action chunk. Since the scorer softmax
also replaces the previous context-attention aggregation, a misaligned scorer
can remove useful mixture information.

The result answers the immediate Gate 3.4b question:

```text
Explicit motion-regret candidate scoring is not enough to make diversity usage
more deployably valuable under the short-budget matched setup.
```

## Decision

Do not expand Gate 3.4c to the full control matrix. The promotion metric did
not clear the first full-aligned check:

```text
Gate 3.4c full aligned temporal MSE: 0.034779
Gate 3.4 full aligned temporal MSE:  0.034262
```

Keep the implemented scorer as a reusable diagnostic, but do not promote it as
the main action-head branch.

Next controlled branch should move to one of:

```text
1. temporal_action_regret scoring, only if another tiny scorer ablation is desired;
2. a small flow/diffusion residual decoder with the same attribution controls.
```

Given this result, the stronger next mainline is the small flow/diffusion
residual decoder, while preserving:

```text
full aligned event/rank/prob
shuffled event metadata
rank/prob-only
mean replacement
context-only/no-prior
same-capacity no-prior decoder
eval-time mean collapse / permutation / batch-mismatch audits
```

## Verification

Completed before the full run:

```text
.venv/bin/python -m pytest tests/test_motion_prior_action_head.py tests/test_predicted_event_mixture_action_head_usage_audit.py
.venv/bin/ruff check src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py tests/test_motion_prior_action_head.py
.venv/bin/python -m compileall -q src/geomoco_wm/models/motion_prior_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/evaluate_predicted_event_mixture_action_head.py
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_4c_sample_score_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_4c_sample_score_top4_k16_seed17/
docs/experiments/plans/2026-06-16_gate3_4c_sample_score_temporal_regret_plan.md
```
