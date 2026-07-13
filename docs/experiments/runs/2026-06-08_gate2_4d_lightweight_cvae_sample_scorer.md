# Gate 2.4d Lightweight cVAE Sample Scorer

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4d
- Purpose: test whether a lightweight deployable scorer can select useful
  future-motion samples from the calibrated Gate 2.4c cVAE without GT
  best-of-K selection at test time.

## Dataset Slice

Source:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 task files | 8 |
| demos | 400 |
| windows | 16,518 |
| context length | 2 |
| horizon | 8 |
| split policy | episode |

## Method

Frozen modules:

```text
Gate 2.4c visual cVAE
Gate 1.6 geodesic ActionDecoder
```

For each window:

```text
condition c_t = cVAE.condition(context, visual, suite_task)
sample K=16 future_motion candidates from p(z | c_t)
action_k = frozen_action_decoder(context, future_motion_k)
score_k = SampleScoreNet(c_t, future_motion_k, action_k)
```

Training target:

```text
target_k = - standardized MSE(action_k, gt_action_chunk)
loss = CE(softmax(target_k), log_softmax(score_k))
```

Test-time readouts:

- prior mean: deployable stable baseline;
- random sample mean: naive stochastic baseline;
- scorer argmax: deployable learned readout;
- scorer soft motion: deployable soft aggregation over future motions;
- oracle best-of-K action: non-deployable upper-bound diagnostic.

## Code Changes

- `src/geomoco_wm/models/sample_readout.py`
  - added `SampleScoreNet`.
- `scripts/train_visual_cvae_sample_scorer.py`
  - trains and evaluates the lightweight cVAE sample scorer.
- `tests/test_future_motion_predictor.py`
  - added ScoreNet shape/validation and listwise ranking-loss tests.

## Commands

Smoke:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4d_smoke_seed7 \
  --num-samples 4 \
  --epochs 1 \
  --batch-size 8 \
  --max-windows 64 \
  --seed 7 \
  --device cpu \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Seed 7:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed7 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Seed 17:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed17 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Artifacts

| seed | metrics | checkpoint |
| ---: | --- | --- |
| 7 | `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed7/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed7/model.pt` |
| 17 | `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed17/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed17/model.pt` |

## Results

| seed | best epoch | prior action MSE | random sample action MSE | scorer argmax action MSE | scorer soft action MSE | oracle best action MSE | top1 oracle match | oracle rank | regret | scorer gripper MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 20 | 0.042666 | 0.042960 | 0.042053 | 0.042505 | 0.038578 | 0.233 | 6.455 | 0.003475 | 0.177693 |
| 17 | 9 | 0.039196 | 0.039406 | 0.038350 | 0.038914 | 0.035213 | 0.244 | 6.774 | 0.003137 | 0.152846 |
| mean | - | 0.040931 | 0.041183 | 0.040201 | 0.040709 | 0.036895 | 0.238 | 6.614 | 0.003306 | 0.165270 |

Additional mean metrics:

| metric | value |
| --- | ---: |
| scorer vs prior action-MSE improvement | 1.78% |
| scorer vs random sample action-MSE improvement | 2.38% |
| prior-to-oracle readout gap closed | 18.09% |
| prior gripper MSE | 0.167670 |
| scorer argmax gripper MSE | 0.165270 |
| prior action translation L2 | 0.014448 m |
| scorer argmax action translation L2 | 0.014234 m |
| prior rotation geodesic | 2.056963 deg |
| scorer argmax rotation geodesic | 2.091603 deg |

## Interpretation

Gate 2.4d passes the minimal deployable-readout test:

```text
prior mean action MSE: 0.040931
random sample action MSE: 0.041183
ScoreNet argmax action MSE: 0.040201
```

The scorer improves over both deployable baselines in both seeds. It also
slightly improves gripper MSE.

But the result is modest:

```text
oracle best-of-K action MSE: 0.036895
ScoreNet argmax action MSE: 0.040201
```

The scorer closes only `18.09%` of the prior-to-oracle readout gap. Top-1
oracle match is also low at `0.238`, and the selected sample is around rank
`6.61` among `K=16` candidates.

## Decision

Promote lightweight ScoreNet as the first working deployable cVAE sample
readout.

Do not yet claim that the readout has solved stochastic planning. The next
branch should improve sample scoring with gripper/contact/executability
signals, or a stronger ranking objective, before moving to multimodal action
heads.
