# Gate 3.1d Predicted Event Mixture

## Purpose

Gate 3.1c proved that oracle event-mode conditioning is a strong upper bound.
Gate 3.1d replaces the oracle event label with a deployable event-mode
prediction route:

```text
visual/proprio context -> event-mode probe -> top-M event modes
top-M event modes -> event-conditioned cVAE -> K future_delta_gripper samples
```

This run is the first predicted-event mixture evaluator. It tests whether the
correct event mode appears in the generated sample set, not whether a final
planner/readout can already choose the right sample.

## Dataset Slice

```text
windows:
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl

visual cache:
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5

event labels:
outputs/event_modes/gate3_1a_event_modes_2files.json
```

The slice has 16,518 windows from the two-file four-suite setup. Evaluation uses
the same episode split policy as the source cVAE checkpoints.

## Code

```text
src/geomoco_wm/data/predicted_event_mixture.py
scripts/evaluate_predicted_event_cvae_mixture.py
tests/test_event_modes.py
```

The event probe was trained on the stable 8 event classes. The event-conditioned
cVAE was trained with all observed classes, including rare mixed-transition
classes. The evaluator maps stable8 probabilities into the cVAE event class set
and gives missing rare classes zero probability.

## Commands

Top-M sweep:

```bash
.venv/bin/python scripts/evaluate_predicted_event_cvae_mixture.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17}/model.pt \
  --output-json outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed{7,17}_top{1,2,4}_k16.json \
  --top-m {1,2,4} \
  --num-samples 16 \
  --batch-size 64 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed{7,17}/model.pt
```

Implementation note: this v1 evaluator uses rank-uniform allocation. With K=16,
top-2 receives 8 samples per rank, and top-4 receives 4 samples per rank.

## Results

Mean across seeds 7 and 17:

| branch | event top-M coverage | prior action MSE | best-of-K action MSE | prior gripper MSE | best-of-K gripper MSE | sample pair L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unconditional joint cVAE | n/a | 0.043816 | 0.022139 | 0.183879 | 0.052857 | 0.612986 |
| shuffled-event cVAE | n/a | 0.042406 | 0.021296 | 0.169936 | 0.049051 | 0.574333 |
| oracle-event cVAE | oracle | 0.018448 | 0.014656 | 0.030296 | 0.010571 | 0.179034 |
| predicted top-1 | 0.813228 | 0.050072 | 0.042023 | 0.242129 | 0.192744 | 0.265841 |
| predicted top-2 | 0.879662 | 0.045272 | 0.027537 | 0.211965 | 0.099296 | 1.510387 |
| predicted top-4 | 0.981789 | 0.042992 | 0.015228 | 0.197773 | 0.021546 | 2.704827 |

Event predictor diagnostics, mean across seeds:

| metric | value |
| --- | ---: |
| top-1 event accuracy | 0.813228 |
| transition binary F1 | 0.608249 |
| transition timing accuracy | 0.541212 |
| top-2 coverage | 0.879662 |
| top-4 coverage | 0.981789 |

## Artifacts

```text
outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed7_top1_k16.json
outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed17_top1_k16.json
outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed7_top2_k16.json
outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed17_top2_k16.json
outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed7_top4_k16.json
outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed17_top4_k16.json
```

## Interpretation

Predicted event modes are useful as a proposal structure, but not yet as an
automatic readout.

Top-1 is too narrow. It has moderate event accuracy, but if the top event mode
is wrong, all K futures are conditioned on the wrong discrete mode.

Top-2 improves coverage and best-of-K, but still loses to the unconditional
sample set.

Top-4 is the important result: it reaches 0.981789 event-mode coverage and
best-of-K action MSE 0.015228, close to the oracle-event best-of-K 0.014656.
That means the predicted-event mixture can place very good futures in the
sample set. However, sample diversity is very large and the prior/readout metric
does not automatically select the right future.

## Decision

Gate 3.1d is a positive sample-space result, not a completed deployment result.

Promote predicted top-4 as the next sample proposal source for a downstream
action head/planner, with a guardrail: do not claim the top-4 mixture itself
solves readout. The next mainline step should evaluate whether the Gate 3
action head can consume the predicted top-4 event mixture better than the
unconditional sample set.
