# Gate 3.0a Motion-Prior-Conditioned Action Head

## Purpose

Test whether a downstream action head can use the full set of sampled
`future_delta_gripper` hypotheses from a frozen joint GeoMoCo-cVAE. This is the
first mainline test of the revised positioning: GeoMoCo-WM provides a visually
grounded multimodal future-motion prior, while action selection/aggregation can
live in a downstream head or planner.

## Dataset

- Windows: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl`
- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Windows: `16,518`
- Motion mode: `future_delta_gripper`
- Context/action horizon: `8`
- Split: episode-level train/validation
- Seeds: `7`, `17`

## Code

- Model: `src/geomoco_wm/models/motion_prior_action_head.py`
- Training: `scripts/train_motion_prior_action_head.py`
- Repeated eval: `scripts/evaluate_motion_prior_action_head.py`
- Tests: `tests/test_motion_prior_action_head.py`

## Model

`MotionPriorActionHead` encodes each future-motion candidate temporally, uses the
context/task token as a query over the sample set, and decodes an action chunk.
The first version does not receive raw DINO features directly; visual
attribution is tested by comparing real visual cVAE samples against shuffled
visual cVAE samples.

## Commands

Seed 7 examples:

```bash
.venv/bin/python scripts/train_motion_prior_action_head.py \
  --input-mode sample_set \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7 \
  --quiet
```

Repeated stochastic eval:

```bash
.venv/bin/python scripts/evaluate_motion_prior_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda \
  --seed 1007
```

The same protocol was run for `context_only`, `gt_future`, real `prior_mean`,
real `sample_set`, and shuffled `sample_set` on seeds `7` and `17`.

## Artifacts

```text
outputs/motion_prior_action_head/gate3_0a_context_only_seed7/
outputs/motion_prior_action_head/gate3_0a_context_only_seed17/
outputs/motion_prior_action_head/gate3_0a_gt_future_seed7/
outputs/motion_prior_action_head/gate3_0a_gt_future_seed17/
outputs/motion_prior_action_head/gate3_0a_prior_mean_real_seed7/
outputs/motion_prior_action_head/gate3_0a_prior_mean_real_seed17/
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed17/
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed7/
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed17/
```

Repeated-eval files:

```text
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/repeated_eval_5pass.json
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed17/repeated_eval_5pass.json
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed7/repeated_eval_5pass.json
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed17/repeated_eval_5pass.json
```

## Metrics

For deterministic modes, use `final_action_metrics` from `metrics.json`.
For stochastic sample-set modes, use 5-pass repeated-eval means.

| mode | seed 7 MSE | seed 17 MSE | mean MSE |
| --- | ---: | ---: | ---: |
| context-only | 0.039253 | 0.035685 | 0.037469 |
| real cVAE prior mean | 0.038547 | 0.035574 | 0.037061 |
| real cVAE sample set, K=16 | 0.038277 | 0.035072 | 0.036675 |
| shuffled cVAE sample set, K=16 | 0.043066 | 0.042200 | 0.042633 |
| GT future upper bound | 0.004421 | 0.004827 | 0.004624 |

Additional mean diagnostics:

| mode | mean translation m L2 | mean rotation geodesic deg | mean gripper MSE |
| --- | ---: | ---: | ---: |
| context-only | 0.013046 | 2.031658 | 0.158123 |
| real cVAE prior mean | 0.012360 | 1.951757 | 0.167009 |
| real cVAE sample set, K=16 | 0.012373 | 1.970024 | 0.164061 |
| shuffled cVAE sample set, K=16 | 0.013382 | 2.060474 | 0.186272 |
| GT future upper bound | 0.006945 | 1.211766 | 0.001530 |

Repeated-eval sample-set MSE std:

| mode | seed 7 std | seed 17 std |
| --- | ---: | ---: |
| real cVAE sample set, K=16 | 0.000440 | 0.000194 |
| shuffled cVAE sample set, K=16 | 0.000218 | 0.000239 |

## Interpretation

Gate 3.0a is a positive first result for the revised motion-prior interface.
The real visual sample-set action head improves over context-only, prior mean,
and shuffled sample-set controls across both seeds. The gain over prior mean is
small, but the shuffled control gap is large enough to show that aligned visual
cVAE samples carry useful downstream action information.

The GT-future upper bound remains far better, so the bottleneck is not solved.
However, this result supports the claim that GeoMoCo-WM can be useful as a
multimodal future-motion proposal prior without requiring the cVAE itself to
select a single best future.

Important measurement note: these numbers are not directly comparable to the
old Gate 2 frozen-action-decoder table. Gate 3.0a trains a new downstream
action head, and the correct comparison is within this table's shared protocol.

## Next Decision

Promote Gate 3.0a as the next mainline branch, but keep it modest:

1. Add a formal comparison summary.
2. Test whether sample aggregation can be strengthened without giving the head
   raw DINO features.
3. Consider K sweeps (`K=4/8/16/32`) or set-pooling variants only after this
   result is documented.
