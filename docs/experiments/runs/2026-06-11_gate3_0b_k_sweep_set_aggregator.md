# Gate 3.0b K Sweep And Set Aggregator Ablation

## Purpose

Gate 3.0a showed that a downstream action head can use real visual cVAE
`future_delta_gripper` sample sets better than context-only, prior mean, and
shuffled sample controls. Gate 3.0b tests whether the remaining gap is mainly
from sample count `K` or from the set aggregation architecture.

## Dataset And Protocol

- Windows: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl`
- Motion mode: `future_delta_gripper`
- Window count: `16,518`
- Split: episode-level
- Seeds: `7`, `17`
- Sample-set metrics: 5-pass repeated validation eval
- Real visual cVAE:
  `outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt`
- Shuffled visual cVAE:
  `outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_shuffled_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt`

## Code

- Model: `src/geomoco_wm/models/motion_prior_action_head.py`
- Train: `scripts/train_motion_prior_action_head.py`
- Eval: `scripts/evaluate_motion_prior_action_head.py`
- Test: `tests/test_motion_prior_action_head.py`

New model options:

```text
--set-aggregator mean_pool | context_attention | multi_query_attention
--set-query-count 4
```

## K Sweep

Aggregator fixed to `context_attention`. K=16 reuses Gate 3.0a repeated-eval
results.

| K | real seed 7 | real seed 17 | real mean | shuffled seed 7 | shuffled seed 17 | shuffled mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.038967 | 0.035365 | 0.037166 | 0.041052 | 0.036684 | 0.038868 |
| 8 | 0.040189 | 0.035354 | 0.037772 | 0.043201 | 0.037121 | 0.040161 |
| 16 | 0.038277 | 0.035072 | 0.036675 | 0.043066 | 0.042200 | 0.042633 |
| 32 | 0.038288 | 0.034841 | 0.036565 | 0.041779 | 0.041413 | 0.041596 |

Best real mean is K=32, but the improvement over K=16 is only `0.000110`
absolute action MSE. K=8 regresses on seed 7. Therefore K is not a strong,
monotonic lever in the current action head.

## Set Aggregator Ablation

K fixed to 16.

| aggregator | real seed 7 | real seed 17 | real mean | shuffled seed 7 | shuffled seed 17 | shuffled mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `context_attention` | 0.038277 | 0.035072 | 0.036675 | 0.043066 | 0.042200 | 0.042633 |
| `mean_pool` | 0.038469 | 0.034912 | 0.036691 | 0.043238 | 0.041532 | 0.042385 |
| `multi_query_attention` | 0.039303 | 0.034596 | 0.036949 | 0.043439 | 0.041227 | 0.042333 |

The aggregators are close on real samples. `mean_pool` nearly matches
`context_attention`, and `multi_query_attention` helps seed 17 but hurts seed 7.
This suggests the current bottleneck is not simply a missing multi-query set
readout.

## Artifacts

K sweep outputs:

```text
outputs/motion_prior_action_head/gate3_0b_k{4,8,32}_contextattn_real_seed{7,17}/
outputs/motion_prior_action_head/gate3_0b_k{4,8,32}_contextattn_shuffled_seed{7,17}/
```

Aggregator outputs:

```text
outputs/motion_prior_action_head/gate3_0b_k16_meanpool_real_seed{7,17}/
outputs/motion_prior_action_head/gate3_0b_k16_multiquery_real_seed{7,17}/
outputs/motion_prior_action_head/gate3_0b_k16_meanpool_shuffled_seed{7,17}/
outputs/motion_prior_action_head/gate3_0b_k16_multiquery_shuffled_seed{7,17}/
```

Each output directory contains:

```text
metrics.json
model.pt
repeated_eval_5pass.json
```

## Interpretation

Gate 3.0b supports the Gate 3.0a conclusion but does not produce a major new
architecture win.

What held:

- real visual motion-prior samples remain better than shuffled controls across
  the main comparisons;
- K=16/K=32 remain the best real settings;
- the downstream action head can use the motion-prior sample set.

What did not hold strongly:

- increasing K is not monotonic;
- K=32 gives only a tiny improvement over K=16;
- multi-query aggregation does not robustly beat context-query attention;
- mean pooling nearly matches attention, so the current head may be extracting
  a coarse sample-set statistic rather than doing rich planning over modes.

## Next Decision

Do not spend the next step on another small aggregator tweak. The more valuable
next branch is to diagnose whether the sample set actually contains distinct
action-useful modes per task/window and whether the action head is ignoring
that diversity.

Suggested Gate 3.0c:

```text
sample-set mode/diversity and action-head usage audit
```

Questions:

```text
Are multiple cVAE samples mapped to meaningfully different action chunks?
Does the action head output change when samples are permuted, dropped, or replaced by the mean?
Which tasks/windows benefit from K=32 over K=16?
Does real-vs-shuffled separation correlate with gripper transitions or long-horizon LIBERO-10 windows?
```
