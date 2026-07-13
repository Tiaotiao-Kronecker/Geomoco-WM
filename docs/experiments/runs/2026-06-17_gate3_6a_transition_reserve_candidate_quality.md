# Gate 3.6a Transition-Reserve Candidate Quality

- Date: 2026-06-17
- Status: completed, transition mechanism-positive but not promoted as default
- Gate: 3.6a
- Purpose: move upstream from residual adapters to predicted-event candidate
  quality while preserving the Gate 3.4 temporal action decoder.

## Motivation

Gate 3.5b/3.5c showed that residual adapters can repair overall action MSE, but
do not fix the transition bottleneck. Gate 3.6a therefore changes only the
predicted-event candidate policy:

```text
default: top-M by event probability
3.6a: reserve one candidate slot for a transition class when transition
      probability is present but absent from top-M
```

The decoder remains the Gate 3.4 `sequence_mlp` temporal action decoder and the
same event/rank/prob sample metadata interface.

## Config

```text
script: scripts/train_predicted_event_mixture_action_head.py
eval: scripts/evaluate_predicted_event_mixture_action_head.py
usage audit: scripts/audit_predicted_event_mixture_action_head_usage.py
event_candidate_policy: transition_reserve
transition_reserve_threshold: 0.15
event_top_m: 4
num_samples: 16
sample_feature_mode: event_rank_prob
temporal_action_decoder_mode: sequence_mlp
temporal_action_loss_weight: 1.0
selection_metric: temporal_action_transition_mse
epochs: 20
batch size: 64
seeds: 7, 17
device: cuda
```

Implementation note: the original `topk` policy remains the default for
backward compatibility. The new policy is explicit in checkpoint metrics.

## Commands

Full aligned:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
    --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
    --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
    --output-dir outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_top4_k16_seed${seed} \
    --event-top-m 4 \
    --num-samples 16 \
    --event-candidate-policy transition_reserve \
    --transition-reserve-threshold 0.15 \
    --sample-feature-mode event_rank_prob \
    --temporal-action-decoder-mode sequence_mlp \
    --temporal-action-loss-weight 1.0 \
    --selection-metric temporal_action_transition_mse \
    --epochs 20 \
    --batch-size 64 \
    --seed ${seed} \
    --device cuda \
    --quiet
done
```

Matched controls used the same config with:

```text
sample_feature_mode=shuffled_event_rank_prob
sample_feature_mode=rank_prob_only
future_input_control=mean_repeated
future_input_control=context_only
```

Important correction: early seed-17 runs were accidentally launched without
`--seed 17`; those artifacts were overwritten by corrected seed-17 runs before
the tables below were computed.

## Results

Mean over seeds 7 and 17, 5-pass repeated eval. Main rows use
`temporal_action_*` metrics because the active decoder is the Gate 3.4 temporal
branch.

| branch | temporal MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.4 top-k reference | 0.034262 | 0.131311 | 0.022542 | 0.149383 |
| full transition-reserve | 0.036391 | 0.126421 | 0.025539 | 0.159475 |
| shuffled event metadata | 0.037033 | 0.127638 | 0.026105 | 0.162219 |
| rank/prob-only metadata | 0.037415 | 0.128118 | 0.026476 | 0.157421 |
| mean repeated | 0.034992 | 0.127759 | 0.023820 | 0.152573 |
| context-only/no-prior | 0.039660 | 0.131626 | 0.028587 | 0.166162 |

Same table for base action output, for completeness:

| branch | MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| full transition-reserve | 0.036272 | 0.126023 | 0.025451 | 0.157966 |
| shuffled event metadata | 0.037242 | 0.128329 | 0.026247 | 0.162774 |
| rank/prob-only metadata | 0.036993 | 0.128208 | 0.025989 | 0.155700 |
| mean repeated | 0.035019 | 0.127929 | 0.023836 | 0.152653 |
| context-only/no-prior | 0.040036 | 0.133721 | 0.028748 | 0.168344 |

## Attribution Ledger

For `temporal_action_mse`:

```text
prior gain    = context-only - full aligned   = +0.003268
metadata gain = shuffled - full aligned       = +0.000641
metadata gain = rank/prob-only - full aligned = +0.001023
diversity gain = mean_repeated - full aligned = -0.001399
vs Gate 3.4 top-k reference                   = -0.002129
```

For `temporal_action_transition_mse`:

```text
prior gain    = context-only - full aligned   = +0.005205
metadata gain = shuffled - full aligned       = +0.001217
metadata gain = rank/prob-only - full aligned = +0.001697
diversity gain = mean_repeated - full aligned = +0.001338
vs Gate 3.4 top-k reference                   = +0.004890
```

Interpretation: transition-reserve candidate quality gives a real transition
gain and the transition gain depends on aligned prior candidates plus metadata.
However, it trades away sustain/overall performance, and the separately trained
`mean_repeated` control is better on overall temporal MSE.

## Usage Audit

Mean over seeds 7 and 17, 3-pass eval-time audit on full aligned:

| eval-time variant | temporal MSE |
| --- | ---: |
| original | 0.036363 |
| mean repeated | 0.042208 |
| rank1 only | 0.047857 |
| subset K=4 | 0.074916 |
| drop rank1 | 0.170306 |
| batch mismatch | 0.305760 |

Additional audit metrics:

```text
delta original vs mean_repeated temporal action L2 = 0.418892
delta original vs permuted temporal action L2      = 0.000000270
sample pair L2                                     = 2.693237
single-sample temporal best-vs-mean gap            = 0.125551
```

The checkpoint uses K-sample structure at runtime. The caveat is still the
matched trained mean-repeated control: runtime diversity is useful inside the
full checkpoint, but training can adapt to mean-only samples and recover better
overall MSE.

## Post-hoc Attribution Correction

Gate 3.6b/3.6c later showed that the `transition_reserve` rule did not trigger
at `top_m=4`: the predicted top-4 event set already contained at least one
transition candidate for every train and validation window. A top-k control
with only `selection_metric=temporal_action_transition_mse` exactly reproduced
the Gate 3.6a metrics.

Therefore this run should no longer be cited as evidence that candidate
reserve/replacement caused the transition gain. The corrected attribution is:

```text
Observed transition gain: real.
Likely cause: transition-MSE checkpoint selection / early stopping.
Not supported: transition_reserve candidate replacement at top_m=4.
```

The controls below still describe the selected checkpoint's dependence on
aligned prior samples, event metadata, and diversity, but they do not prove a
new upstream candidate-allocation mechanism.

## Interpretation

Gate 3.6a is a controlled transition-selection positive for the transition
bottleneck:

```text
transition MSE improves from Gate 3.4 0.131311 to 0.126421;
context-only/no-prior cannot match it;
shuffled/rank-prob-only metadata controls are weaker;
mean_repeated is also weaker on transition.
```

But it is not promoted as the default because:

```text
overall temporal MSE worsens from 0.034262 to 0.036391;
sustain MSE worsens from 0.022542 to 0.025539;
gripper MSE is also slightly worse than Gate 3.4.
```

The branch validates the upstream direction without yet solving the overall
policy trade-off.

## Artifacts

```text
outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_shuffled_event_top4_k16_seed*/
outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_rankprob_top4_k16_seed*/
outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_mean_repeated_top4_k16_seed*/
outputs/motion_prior_action_head/gate3_6a_transition_reserve_t015_context_only_seed*/
```

## Next Decision

Do not scale residual adapters next. Also do not continue fixed-threshold
`transition_reserve` tuning at `top_m=4` unless the policy is changed so it
actually modifies the selected candidate set. Cleaner next steps are:

```text
1. treat transition-based checkpoint selection as a diagnostic Pareto lever;
2. design a candidate policy that actually changes top-4 composition, such as
   transition-timing diversity or multiple transition slots;
3. score candidate futures by downstream action/transition regret before they
   are forced into the sample set;
4. keep the same decoder/prior/metadata/diversity controls.
```
