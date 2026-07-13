# Gate 3.1g Event Metadata Ablation

## Purpose

Gate 3.1f showed that event-aware sample consumption improves the action head.
Gate 3.1g isolates which part of the per-sample event metadata carries the
gain.

First-principles questions:

```text
1. Event identity: does the action head need to know which event mode generated
   each future sample?
2. Event confidence: does the action head need event rank/probability to judge
   which samples are more context-compatible?
3. Alignment: does the event identity have to be correctly aligned with the
   sample, or is this just extra model capacity?
```

## Ablations

All branches use the same predicted top-4 event-conditioned cVAE sample set.
Only the metadata given to the action head changes.

| branch | sample metadata |
| --- | --- |
| anonymous top-4 | none |
| event-only | event-mode one-hot |
| rank/prob-only | event rank + normalized top-M probability |
| full event/rank/prob | event-mode one-hot + rank + probability |
| shuffled-event/rank/prob | batch-shuffled event one-hot + aligned rank/probability |

## Code

```text
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
src/geomoco_wm/models/motion_prior_action_head.py
tests/test_motion_prior_action_head.py
```

## Commands

Training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17}/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_1g_<mode>_top4_k16_seed{7,17} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode <event_only|rank_prob_only|shuffled_event_rank_prob> \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden-dims 512,512 \
  --token-dim 256 \
  --num-heads 4 \
  --temporal-layers 1 \
  --set-aggregator context_attention \
  --dropout 0.1 \
  --seed {7,17} \
  --device cuda \
  --quiet
```

Repeated evaluation:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_1g_<mode>_top4_k16_seed{7,17}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_1g_<mode>_top4_k16_seed{7,17}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

## Results

Mean across seeds 7 and 17, 5-pass stochastic evaluation:

| branch | action MSE | action MAE | translation m MSE | rotation geodesic deg | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.0 real sample-set K=16 | 0.036675 | 0.088318 | 0.000072 | 1.970024 | 0.164061 |
| Gate 3.1e anonymous top-4 | 0.038024 | 0.094724 | 0.000077 | 1.997687 | 0.167432 |
| Gate 3.1g event-only | 0.037108 | 0.092792 | 0.000075 | 1.964412 | 0.163236 |
| Gate 3.1g rank/prob-only | 0.036069 | 0.089186 | 0.000073 | 1.970533 | 0.158481 |
| Gate 3.1g shuffled-event/rank/prob | 0.036228 | 0.091946 | 0.000077 | 2.007828 | 0.154256 |
| Gate 3.1f full event/rank/prob | 0.034767 | 0.089716 | 0.000072 | 1.968135 | 0.150052 |

## Interpretation

The full event/rank/prob metadata is necessary for the strongest result.

Event identity alone is not enough:

```text
anonymous top-4: 0.038024
event-only:      0.037108
```

Rank/probability alone is useful and beats the Gate 3.0 real sample-set
baseline:

```text
rank/prob-only: 0.036069
Gate 3.0 real: 0.036675
```

But full metadata is better:

```text
full event/rank/prob: 0.034767
rank/prob-only:       0.036069
```

Shuffling event identity while keeping rank/probability aligned does not match
the full branch:

```text
shuffled-event/rank/prob: 0.036228
full event/rank/prob:     0.034767
```

So the gain is not just extra dimensions or event-prior confidence. The action
head benefits when event identity is aligned with each sample and paired with
rank/probability confidence.

## Decision

Promote full event/rank/prob metadata as the current Gate 3.1 interface.

Next mainline:

```text
Gate 3.2: scale or stress-test the promoted event-aware interface
```

Do not move to flow matching until this deterministic interface is tested on a
broader suite/task slice or on harder long-horizon subsets.
