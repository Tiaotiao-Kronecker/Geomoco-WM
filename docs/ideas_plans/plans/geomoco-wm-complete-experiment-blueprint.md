# GeoMoCo-WM Complete Experiment Blueprint

- Date: 2026-06-05
- Project: `Geomoco-WM`
- Status: planning archive

## Core Decision

`GeoMoCo-WM` should be designed as a visual-grounded future-motion prior, not as
a VAE-only extension of old GeoMoCo.

The first serious route is:

```text
RGB history + proprioception + task
  -> DINO-grounded visual context
  -> GeoMoCo-cVAE future geometric motion prior
  -> predictive state/progress gates
  -> shared action decoder
  -> closed-loop validation only after offline gates pass
```

The project should not claim full world-model value until future motion is
predicted before observation and shown to help a controlled decoder or rollout.

## Lessons From Previous GeoMoCo Experiments

Old evidence from `/home/user/projects/se3-group-motion-control` shapes the new
design.

### What Failed Or Was Ambiguous

- Pure GeoMoCo latent was not enough for policy/retrieval success.
- Frozen GeoMoCo latent sometimes improved offline action MSE but failed or hurt
  closed-loop control.
- DINO visual grounding was useful, but pure DINO often became a strong baseline.
- Raw weighted sums of DINO distance and GeoMoCo latent distance were brittle.
- Learned-linear RGB grounding showed low training error but destroyed
  executable-neighbor order.
- Task7 often saturated under DINO/EE+gripper grounding, hiding marginal gains.
- Task5 showed that DINO pools contained future-phase candidates, but GenB/ranker
  and executor logic could not reliably convert them into better actions.
- Oracle or deployable progress/phase features improved candidate selection,
  implying that phase/progress estimation is a first-class bottleneck.

### Main Implication

Do not repeat the old pattern:

```text
DINO feature distance + GeoMoCo latent distance -> retrieval score
```

Instead, use DINO as a visual context source for a learned future-motion prior:

```text
DINO tokens -> task/EEF-conditioned grounding token g_t
g_t + proprio/history/task -> p(z_future)
z_future -> future geometry/progress/action interface
```

## Research Claim Ladder

The project should climb this ladder in order:

1. **Grounding claim**
   DINO-conditioned features predict future geometric progress better than
   proprio/history alone.
2. **Future-motion prior claim**
   GeoMoCo-cVAE improves future-motion coverage over deterministic AE and direct
   residual predictors.
3. **Executable-interface claim**
   Oracle future motion improves a controlled action decoder over direct BC.
4. **Learned-prior action claim**
   Predicted/sampled GeoMoCo future motion recovers part of the oracle benefit.
5. **Long-horizon composition claim**
   Gains are larger on LIBERO-Long / LIBERO-10 tasks that require multiple
   composable phases.

Do not promote a higher claim before the lower gates pass.

## Dataset Design

### Source Suites

Use a staged suite design:

```text
Mechanism tasks:
  task0 drawer, task5 plate, task7 stove or equivalent phase-rich tasks

Main long-horizon suite:
  LIBERO-Long / LIBERO-10-style tasks

Generalization:
  held-out init states, held-out task splits, and held-out object arrangements
```

Task selection criteria:

- oracle/demo replay succeeds reliably;
- direct visual BC is not saturated;
- direct visual BC is not complete floor;
- task has meaningful phases such as reach, contact, move, release, recover;
- future motion matters beyond current-frame visual similarity.

### Raw Episode Fields

Each raw episode should preserve:

```text
episode_id
task_id / task_text
timestep
agentview RGB
wrist RGB when available
robot proprioception
EEF pose
gripper state
action
success / done when available
sim state or object_state teacher when available
```

Object state is a teacher/diagnostic/upper-bound field. The deployable main
model should not require object state at inference.

### Window Contract

Each training item is a temporal window:

```text
context:
  RGB frames [t-H_rgb+1, ..., t]
  proprio/EEF/gripper history [t-H_hist+1, ..., t]
  action history [t-H_hist+1, ..., t-1]
  task id/text

future targets:
  EEF SE(3) delta from t to t+k
  per-step EEF delta sequence
  geometric progress / phase
  future DINO feature or visual token target
  optional object/contact/progress teacher
  action chunk [t, ..., t+H_a-1]

metadata:
  task suite
  episode split
  init state id when available
  oracle replay eligibility
```

Recommended horizons:

```text
RGB history: 2 / 4 frames
motion history: 8 / 16 steps
future motion: 4 / 8 / 16 steps
action chunk: 8 steps
```

### Feature Cache

DINO features should be cached before model training:

```text
cache key:
  suite, task, episode, timestep, camera, DINO model, image preprocessing hash

values:
  CLS/global token
  patch tokens
  optional pooled multi-view token
```

Feature-cache audits:

- image nonblank check;
- feature finite check;
- feature shape/version check;
- RGB/action/proprio alignment check;
- temporal order shuffle control.

### Splits

Use explicit splits:

```text
train:
  model fitting

search/dev:
  choose gates, alpha, feature settings, and decoder family

held-out validation:
  one-shot confirmation after gates pass
```

Never tune alpha, latent weight, task selection, or decoder settings on the
held-out validation split.

## Network Architecture

### Overview

```text
RGB history
  -> frozen DINO
  -> temporal visual grounding module
  -> g_t

proprio/history/task
  -> structured context encoder
  -> r_t

[g_t, r_t]
  -> GeoMoCo-AE / GeoMoCo-cVAE future-motion prior
  -> future motion targets

[g_t, r_t, future motion]
  -> shared action decoder
  -> action chunk
```

### Visual Grounding Module

Inputs:

```text
DINO patch tokens from agentview/wrist
EEF pose, gripper, proprio, task embedding
```

First-pass design:

```text
query = MLP(EEF, gripper, proprio, task)
keys/values = DINO patch tokens over time
attention(query, keys, values) -> g_t
temporal transformer or token-difference block -> g_t_final
```

Required controls:

- global DINO token only;
- no temporal module;
- shuffled temporal order;
- proprio/history only;
- frozen random visual tokens.

### Structured Context Encoder

Inputs:

```text
EEF SE(3) history as Lie-log deltas
gripper history
action history up to t-1
task embedding
optional proprio/joint state
```

Output:

```text
r_t: structured robot-side context token
```

### GeoMoCo-AE Baseline

Purpose:

```text
deterministic future-motion representation
```

Interface:

```text
encoder([g_t, r_t, future_motion]) -> z_future
decoder([g_t, r_t, z_future]) -> future targets
```

### GeoMoCo-cVAE Prior

Training:

```text
posterior q(z | g_t, r_t, future_motion)
prior     p(z | g_t, r_t)
decoder   D(g_t, r_t, z) -> future targets
```

Inference:

```text
z ~ p(z | g_t, r_t)
decoder -> future motion proposal
```

Targets:

- future EEF SE(3);
- per-step EEF delta;
- geometric progress / phase;
- future DINO feature;
- optional object/contact teacher;
- optional GeoMoCo composition latent.

Losses:

```text
L_se3
L_progress
L_future_visual
L_kl
L_comp
L_rank / phase contrastive loss
L_contact_teacher optional
```

### Composition Path

Keep an explicit composition route:

```text
z_A = observed/history motion latent
z_B = predicted next-segment latent
z_AB = compose(z_A, z_B)
```

Also train direct residual future prediction:

```text
z_AB_direct = direct_future_head(g_t, r_t)
```

The direct residual baseline is required because previous results showed it can
be very strong.

### Action Decoder

Gate order:

1. action-chunk MLP/Transformer;
2. Diffusion-Policy-style decoder;
3. MeanFlow/flow-style decoder if the simple decoder shows future-motion value.

Controlled inputs:

```text
direct BC:
  [g_t, r_t] -> action chunk

oracle future motion:
  [g_t, r_t, future_motion_GT] -> action chunk

AE:
  [g_t, r_t, future_motion_AE] -> action chunk

cVAE:
  [g_t, r_t, future_motion_sampled or best-of-K] -> action chunk
```

If oracle future motion does not improve over direct BC, do not promote the
cVAE-to-action path.

## Experiment Sequence

### Gate -1: Task Mining And Headroom Audit

Goal:

```text
choose tasks where future motion can matter
```

Run:

- direct visual BC;
- pure DINO retrieval/action proxy if available;
- oracle replay;
- quick success/floor/saturation audit.

Promote tasks where:

- direct baseline is not saturated;
- direct baseline is not complete floor;
- task has composable phases;
- oracle/demo replay succeeds.

### Gate 0: Data Export And Feature Cache

Deliverables:

- raw episode manifest;
- window manifest;
- DINO feature cache;
- alignment audit report.

Stop if:

- RGB/proprio/action alignment is ambiguous;
- DINO cache is not reproducible;
- oracle action replay fails.

### Gate 1: Visual Grounding Probe

Question:

```text
does visual context help predict future motion/progress?
```

Baselines:

- proprio/history/task only;
- DINO global token;
- DINO patch attention;
- DINO temporal module;
- shuffled DINO temporal order.

Metrics:

- future EEF SE(3) error;
- future progress error;
- future DINO feature error;
- object/contact teacher error if available;
- top-K future-phase candidate coverage.

### Gate 2: Oracle Future-Motion Action Decoder

Question:

```text
if the action decoder receives ground-truth future motion, does it beat direct BC?
```

Compare:

- `[g_t, r_t] -> action`;
- `[g_t, r_t, GT future motion] -> action`.

Promote only if GT future motion clearly improves action prediction or
closed-loop search performance. This is the early stop gate for the action route.

### Gate 3: Future-Motion Prior

Question:

```text
does cVAE improve future-motion coverage over deterministic and direct baselines?
```

Compare:

- AE;
- cVAE prior mean;
- cVAE sampled K;
- direct residual future;
- random latent;
- shuffled latent;
- oracle future motion.

Metrics:

- reconstruction error;
- min-of-K coverage;
- calibration/diversity;
- progress/phase precision;
- composition closure;
- future visual feature prediction.

### Gate 4: Shared Action Decoder

Question:

```text
does predicted future motion help actions under the same decoder?
```

Compare:

- direct BC;
- direct residual future;
- AE future motion;
- cVAE future motion;
- cVAE best-of-K;
- random/shuffled latent;
- oracle future motion.

Promote only if learned future motion closes a meaningful part of the oracle
future-motion gap.

### Gate 5: Mechanism Closed Loop

Use mechanism tasks first:

- drawer/contact task;
- plate push/alignment task;
- stove/knob phase task.

Report:

- success;
- fixed/regressed cases;
- action trace deltas;
- phase/progress trajectory;
- neighbor/action feasibility if using retrieval;
- video snippets or frame sheets when useful.

### Gate 6: LIBERO-Long / LIBERO-10 Main Evaluation

Use long-horizon tasks after Gates 1-5 pass.

Main claim:

```text
GeoMoCo-WM helps most when tasks require multiple composable motion phases.
```

Report:

- success and sample efficiency;
- phase transition accuracy;
- multi-step progress prediction;
- degradation with horizon;
- per-task breakdown;
- oracle gap closure.

### Gate 7: Deferred Visual-Motion Baselines

Only after the DINO route has a clean signal:

- ZipMotion-style visual motion front-end;
- AMPLIFY-style actionless motion prior plus inverse dynamics;
- Fast-WAM-style future visual feature co-training.

Do not make these first-pass dependencies.

## Baseline Matrix

Required baselines:

```text
direct DINO/proprio/task BC
proprio/history/task only
DINO-only predictor
GeoMoCo-AE
GeoMoCo-cVAE
direct residual future latent
random latent
shuffled latent
oracle future motion
GT object-state teacher upper bound
```

Optional strong baselines:

```text
Diffusion Policy action decoder
MeanFlow-style action decoder
ZipMotion front-end
AMPLIFY-style motion-token prior
```

## Metrics

Predictive:

- future EEF SE(3) error;
- future progress RMSE;
- future visual feature error;
- contact/object-state teacher error;
- min-of-K cVAE coverage;
- sample diversity and calibration.

Composition:

- `T_A T_B` vs `T_AB`;
- `compose(z_A, z_B)` vs `z_AB`;
- phase-near hard-negative ranking;
- direct residual baseline gap.

Action:

- action MSE/RMSE/L1;
- action chunk consistency;
- gripper disagreement;
- oracle future-motion action gap;
- closed-loop success;
- fixed/regressed case count.

Grounding:

- DINO pool future-phase coverage;
- top1 near-target phase;
- neighbor overlap/Jaccard to stable baseline;
- action feasibility of selected candidates.

## Promotion And Stop Rules

Promote if:

- visual grounding beats proprio/history on future prediction;
- oracle future motion beats direct BC;
- cVAE improves coverage over AE/direct residual on non-saturated tasks;
- learned future motion closes part of the oracle action gap;
- long-horizon tasks show larger gains than short tasks.

Stop or redesign if:

- DINO/proprio direct policy is saturated on all selected tasks;
- oracle future motion does not help the action decoder;
- random/shuffled latents match GeoMoCo;
- direct residual future latent dominates cVAE everywhere;
- closed-loop gains appear only on tuned search splits and disappear on
  held-out validation.

## Immediate Implementation Checklist

1. Implement visual-grounded dataset exporter.
2. Implement DINO feature cache with alignment audits.
3. Add task mining/headroom scripts.
4. Add visual grounding probe model.
5. Add oracle future-motion action-decoder baseline.
6. Extend GeoMoCo-cVAE context with `g_t`.
7. Add direct residual future-latent baseline.
8. Add shared action decoder and gate reports.
9. Add mechanism closed-loop scripts.
10. Promote to LIBERO-Long / LIBERO-10 only after gates pass.

## 2026-06-08 Mainline Update

The current oracle future-motion branch is an upper-bound diagnostic, not a
complete action oracle:

```text
context + GT future EEF delta -> action decoder -> action chunk
```

It can still be improved by adding GT gripper/contact, future visual/object
change, and stronger inverse-dynamics decoders. This becomes an Oracle v2
calibration side-track.

The mainline order is:

```text
Gate 2.2c visual controls
  -> Gate 2.3 action-aware visual future-motion prior
  -> Gate 2.3b action-aware lambda selection
  -> Gate 2.4 multimodal / stochastic future-motion prior
  -> GeoMoCo-cVAE with validated visual-action route
  -> Oracle v2 upper-bound calibration if needed
```

Gate 2.3a has started with:

```text
motion_loss = MSE(pred_future_ee_delta, gt_future_ee_delta)
action_loss = MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)
total_loss = motion_loss + 0.01 * action_loss
```

This tests whether future motion is not only geometrically accurate but also
action-executable.

Gate 2.3b completed the first action-aware lambda sweep:

```text
lambda 0.000, Gate 2.2b action MSE: 0.049547
lambda 0.003 action MSE: 0.045790
lambda 0.010 action MSE: 0.043174
lambda 0.030 action MSE: 0.042090
```

`lambda_action=0.030` is the current action-value-prioritized default because
it reaches `69.26%` direct-to-oracle action-MSE gap closure. Its caveat is that
future-motion translation L2 worsens from `0.016763` at lambda `0.010` to
`0.018767`, so lambda `0.010` remains the balanced geometry reference.

The next branch should keep both values in view:

```text
primary: lambda_action = 0.030
reference: lambda_action = 0.010
```

The immediate next research target is a multimodal / stochastic future-motion
prior, with step-wise visual conditioning and gripper/contact diagnostics
reported before any cVAE result is promoted as a policy-relevant module.

Gate 2.4a tested step-wise / multi-query visual attention:

```text
base query + step embedding[k] -> query_k
query_k attends DINO patch tokens -> g_{t,k}
[context, g_{t,k}, step embedding[k]] -> future_delta_ee[k]
```

The result is mixed:

```text
single-query lambda 0.030 action MSE: 0.042090
stepwise-query lambda 0.030 action MSE: 0.042687
single-query lambda 0.030 future trans L2: 0.018767
stepwise-query lambda 0.030 future trans L2: 0.017155
```

So step-wise querying helps motion-space translation consistency, but does not
beat the single-query action-value default. The default deterministic branch
therefore remains `cross_attention + lambda_action=0.030`, and the next
mainline should move to stochastic / multimodal future-motion modeling rather
than another deterministic query-only ablation.

Gate 2.4b implemented the first visual-conditioned cVAE:

```text
base = [proprio, suite_task one-hot]
base -> query
query attends DINO patch tokens -> g_t
c_t = [base, g_t]
posterior q(z | c_t, future_motion)
prior     p(z | c_t)
decoder   Dec(c_t, z) -> future_motion
```

The deployable prior-mean result is a weak positive:

```text
deterministic single-query lambda 0.030 action MSE: 0.042090
visual cVAE prior-mean action MSE: 0.041579
deterministic single-query gripper MSE: 0.174519
visual cVAE prior-mean gripper MSE: 0.165615
```

However, the KL is near zero:

```text
mean KL: 0.000740
```

Posterior and prior reconstructions are nearly identical, so this first cVAE
run should not yet be claimed as a meaningful multimodal world-motion model. It
is a promising entry point, but the next gate must calibrate sample coverage,
KL/free-bits or beta schedules, and gripper/contact diagnostics.

Gate 2.4c completed the cVAE stochasticity calibration:

```text
branch: visual cVAE with free_bits=0.02
beta schedule: 0.0 -> 0.001 over 5 epochs
K: 16 prior samples
```

The result is positive for latent usage and coverage:

```text
Gate 2.4b raw KL: 0.000740
Gate 2.4c raw KL: 0.442097

Gate 2.4b sample variance: 0.00000167
Gate 2.4c sample variance: 0.00004481

Gate 2.4b best-of-K motion MSE: 0.000744
Gate 2.4c best-of-K motion MSE: 0.000552

Gate 2.4b best-of-K action MSE: 0.039526
Gate 2.4c best-of-K action MSE: 0.036894
```

The deployable prior mean also improves slightly:

```text
Gate 2.4b prior-mean action MSE: 0.041579
Gate 2.4c prior-mean action MSE: 0.040931
```

But random sample mean is still worse than the prior mean:

```text
Gate 2.4c prior-mean action MSE: 0.040931
Gate 2.4c sample-mean action MSE: 0.041199
```

So the mainline conclusion is:

- cVAE latent collapse has been meaningfully reduced;
- the future-motion distribution contains better action-relevant samples;
- best-of-K is still an oracle diagnostic, not a deployable method;
- the next gate should add a deployable sample scorer/readout and
  gripper/contact diagnostics before claiming multimodal policy value.

The scorer/readout direction has direct precedent in published work, but not as
a single identical module:

- BCQ uses a generative candidate model plus a Q-value selector;
- IBC learns an energy over observation-action pairs and selects low-energy
  actions;
- Trajectory Transformer uses beam/search readout over trajectory candidates;
- Diffuser and QGPO use reward/energy guidance to steer generative sampling;
- Visual MPC / PETS score candidate futures with task costs;
- ACT commonly falls back to prior mean at inference, which is a baseline but
  not a sample selector.

Therefore Gate 2.4d should keep the first implementation simple:

```text
freeze calibrated cVAE
sample K future motions
decode each through frozen action decoder
learn ScoreNet(context, vision, future_motion, decoded_action)
select or soft-aggregate candidates without GT labels at test time
```

The formal plan is:

```text
docs/experiments/plans/2026-06-08_gate2_4d_cvae_sample_readout_plan.md
```

This readout gate is the minimal world-model planning problem, not a side
branch. A full stochastic world model samples multiple future states, videos, or
latent rollouts and then scores them before acting. GeoMoCo-WM currently does
the same at a narrower abstraction level:

```text
vision + proprio + task
  -> sample multiple future EEF SE(3) motion rollouts
  -> score/read out one or aggregate several
  -> action decoder
  -> action chunk
```

The action head should remain deterministic for Gate 2.4d so attribution stays
clean:

```text
Stage A:
  multimodal future-motion cVAE
  deterministic action decoder
  lightweight sample scorer/readout

Stage B:
  add gripper/contact/executability scoring

Stage C:
  compare deterministic action decoder with diffusion / flow / MeanFlow action
  heads

Stage D:
  full multi-rollout planning over future motion, future state/vision, action
  candidates, and value/success scoring
```

If the action head is made multimodal too early, improvements become hard to
attribute: they may come from GeoMoCo future-motion samples, or simply from a
stronger action generator. Therefore the next mainline implementation should be
the lightweight ScoreNet readout first.

### PointWorld Contrast: Where Multimodality Lives

The PointWorld comparison clarifies the mainline narrative. PointWorld-style
multi-rollout control should be treated primarily as planner-side action
candidate rollout: propose or optimize multiple action sequences, roll each
sequence forward through an action-conditioned 3D point-flow world model, score
the predicted future states, and execute the best sequence in an MPC loop.

GeoMoCo-WM should keep a different source of diversity explicit:

```text
PointWorld-style planning:
  action diversity -> multiple predicted world rollouts

GeoMoCo-WM:
  latent future-motion diversity -> multiple future-motion hypotheses
```

Therefore GeoMoCo-WM should not be narrated as a dense 3D PointWorld-like world
model. The sharper claim is a visual-grounded, geometry-structured,
multimodal future-motion interface for action. PointWorld is a useful reference
for the need to score rollouts, but the current mainline should first solve
readout over `future_delta_ee + future_gripper/event` samples before adding an
outer action-sequence MPC loop or a multimodal diffusion/flow action head.

The archived discussion is:

```text
docs/agent_qa/2026-06-10-pointworld-vs-geomoco-wm-multimodal-rollouts.md
```

### Positioning Advantage vs Nearby Work

The project should also avoid claiming that latent multi-rollout generation is
itself new. PlaNet/Dreamer, stochastic video prediction, diffusion trajectory
planners, ACT-style CVAE action heads, and BCQ-style candidate generators all
contain versions of latent or noise-conditioned future/action sampling.

GeoMoCo-WM's sharper advantage should be framed as the choice of rollout space:

```text
not pixels
not generic latent state
not direct action chunk
but action-relevant future_delta_ee + future_gripper/event
```

This gives a compact and inspectable interface where visual grounding, EEF
geometry, gripper/open-close timing, sample readout, and action decoding can be
measured separately. Therefore Gate 2.5d is not merely another scorer
experiment; it is the first deployable test of whether the useful
`best-of-K=0.022139` joint cVAE sample coverage can be converted into a
non-GT-selected action interface.

The archived discussion is:

```text
docs/agent_qa/2026-06-10-geomoco-wm-positioning-advantages-vs-related-work.md
```

Gate 2.4d completed the first lightweight ScoreNet readout:

```text
ScoreNet(c_t, future_motion_k, decoded_action_k) -> score_k
target: action-distance ranking
K: 16
```

Mean result:

```text
prior mean action MSE: 0.040931
random sample action MSE: 0.041183
ScoreNet argmax action MSE: 0.040201
oracle best-of-K action MSE: 0.036895
```

This is a positive but modest readout result:

- ScoreNet beats prior mean by `1.78%`;
- ScoreNet beats random sample mean by `2.38%`;
- ScoreNet closes `18.09%` of the prior-to-oracle readout gap;
- gripper MSE improves slightly from `0.167670` to `0.165270`;
- top-1 oracle match is only `0.238`.

The next mainline should strengthen the readout with gripper/contact or
executability signals and harder ranking objectives before adding a multimodal
action head.

Gate 2.4e/2.4f/2.4g refined this readout bottleneck:

```text
Gate 2.4e:
  naive SE(3) / SE(3)+gripper scorer targets
  -> diagnostic value, but worse action MSE than flat ScoreNet

Gate 2.4f:
  SE(3)-oracle and gripper-aware oracle rank evaluation
  -> structured scorers improve their own rank but still lose on deployable
     action MSE

Gate 2.4g:
  naive SE(3)+gripper hard-negative auxiliary loss
  -> negative ablation; structured-score negatives are not reliable enough
```

The resulting mainline adjustment is Gate 2.4h:

```text
GeoMoCo Phase/Event Probe
```

This is not a gripper trick. It is a probe for the core GeoMoCo claim:

```text
manipulation success depends on phase/composition timing:
  approach -> align -> close/contact -> transport -> release
```

The key question becomes:

```text
Do visual-grounded GeoMoCo future-motion samples express the correct
manipulation phase transitions, and can the readout identify them?
```

Gate 2.4h should proceed in four small steps:

```text
2.4h-a GT gripper/event label audit
  derive close/open/hold/mixed labels from future action chunks;
  infer gripper close sign from action-vs-gripper-width deltas when possible;
  report event timing distributions by suite/task;
  detect task/time shortcut risk.

2.4h-b visual phase/event probe
  compare task/proprio only, visual only, future motion only,
  proprio+future motion, visual+proprio, visual+proprio+future motion,
  and shuffled visual controls for event timing prediction.

2.4h-c cVAE sample event-alignment analysis
  decode each sampled future motion through the frozen action decoder;
  derive candidate event labels;
  test whether event-aligned samples are closer to oracle best-of-K.

  Completed result:
    event oracle best-of-K improves event accuracy from 0.836666 to 0.849049,
    but transition step within 1 only reaches 0.219359 and ScoreNet reaches
    0.175082. Therefore cVAE samples contain limited event-aligned candidates,
    while the current flat ScoreNet does not reliably select close/open
    transition timing.

2.4h-d event-aware ScoreNet only if probes are positive
  keep flat action-MSE ranking as the promotion metric;
  add event/timing auxiliary losses only after audit/probe controls pass.

  Completed result:
    weak event alignment rank loss (weight 0.1) is not promoted. It changes
    action MSE from 0.040201 to 0.040255 and transition step-within-1 from
    0.175082 to 0.173790. Therefore the bottleneck is not solved by adding
    event loss to the readout head.

2.4i event-fidelity interface audit
  move upstream from readout engineering;
  test whether future EEF-only motion plus frozen action decoder has enough
  information to represent gripper close/open timing;
  consider adding future gripper/event channels or an event-aware action
  decoder diagnostic before returning to stronger sample readouts.

  Completed result:
    EEF-only future motion action MSE is 0.031474 with gripper MSE 0.184683.
    EEF+oracle future gripper action MSE is 0.004202 with gripper MSE
    0.000241. This is an 86.65% action-MSE reduction and a 99.87% gripper-MSE
    reduction over EEF-only. Transition-event accuracy also jumps from
    0.784005 to 1.000000. Therefore the missing channel is gripper/event, not
    only readout capacity.

2.5a visual future-gripper/event predictor
  predict future gripper commands or transition/event channels from
  visual/proprio/task context;
  evaluate gripper MSE and transition event metrics before combining with EEF.

  Completed result:
    real DINO patchpool visual grounding predicts future gripper/event timing
    much better than task/proprio and shuffled visual controls. Mean gripper
    MSE is 0.172088 for real visual, 0.233726 for shuffled visual, and
    0.324415 for task/proprio. Transition accuracy is 0.634542 for real
    visual, 0.481249 for shuffled visual, and 0.014386 for task/proprio.
    In the bridge diagnostic, GT future EEF + predicted visual gripper reaches
    action MSE 0.028987 and gripper MSE 0.173254, improving over the Gate 2.4i
    EEF-only oracle interface (0.031474 action MSE, 0.184683 gripper MSE) but
    remaining far from the GT future-gripper upper bound (0.004202 action MSE).
    Therefore the gripper/event channel is predictably useful, but the next
    gate must remove the remaining GT future-EEF privilege.

2.5b visual future-EEF+gripper predictor
  jointly predict future_delta_ee + future_gripper/event;
  feed predicted joint representation into the action decoder.

  Modular bridge diagnostic completed:
    separately composed predicted EEF + predicted gripper preserves visual
    attribution but is not promoted. Mean action MSE is 0.050333 for real
    visual, 0.065466 for shuffled visual, and 0.079633 for task/proprio, so
    aligned visual grounding still matters. However it does not beat the
    previous best EEF-only learned prior action MSE 0.042090. Decomposition
    shows GT EEF + predicted gripper reaches 0.028987, predicted EEF + GT
    gripper reaches 0.025443 with poor SE(3) MSE 0.029632, and predicted EEF +
    predicted gripper reaches 0.050333. Therefore separate EEF and gripper
    predictors compound noise inside the joint decoder.

  Next 2.5b sub-step:
    train a deterministic joint future_delta_gripper predictor directly, with
    optional action-aware loss through the future_delta_gripper action decoder.
    Only promote to cVAE after the deterministic joint predictor beats the
    modular bridge and preferably recovers the best EEF-only learned prior.

  Joint predictor completed:
    lambda_action 0.030 improves over the modular bridge only slightly
    (0.050333 -> 0.049103). Increasing action-aware weight to 0.300 fixes the
    loss-scale mismatch and reaches action MSE 0.040688, beating the previous
    best EEF-only learned prior 0.042090. Controls pass: task/proprio is
    0.084648 and shuffled visual is 0.063790. Event metrics also pass:
    transition accuracy is 0.560270 for real visual, 0.330051 for shuffled
    visual, and 0.009672 for task/proprio. Therefore future_delta_ee +
    future_gripper/event is promoted as the deterministic output space.

2.5c GeoMoCo-cVAE with EEF+gripper/event output
  upgrade stochastic world-motion samples from EEF-only to manipulation-event
  aware rollouts. Compare against the Gate 2.5b-joint deterministic visual
  baseline, not only against the older EEF-only future-motion baselines.

  Completed result:
    joint cVAE prior mean is visually grounded but not yet deployable as an
    improvement over deterministic joint. With real visual patchpool,
    prior_recon_weight 0.5, free_bits 0.02, and lambda_action 0.300, prior mean
    action MSE is 0.043816 versus deterministic joint 0.040688. However
    best-of-K action MSE is 0.022139, showing strong latent sample coverage.
    Real visual beats shuffled visual on prior action MSE (0.043816 vs
    0.068816), best-of-K action MSE (0.022139 vs 0.030576), and transition
    accuracy (0.571094 vs 0.223375). Therefore the joint cVAE sample space is
    promoted, but raw prior mean is not.

2.5d joint cVAE sample readout/scorer
  convert the large joint cVAE best-of-K gap into a deployable readout;
  compare against deterministic joint action MSE 0.040688 and cVAE prior mean
  action MSE 0.043816;
  keep event fidelity and shuffled visual controls mandatory.
```

Visual grounding should be used as current-state evidence, not as an event
label source:

```text
DINO visual tokens answer:
  where is the object?
  is the gripper aligned?
  has the drawer/container/state changed?

future motion answers:
  what phase transition is the sample proposing?

gripper/action labels answer:
  what observable manipulation event happened in the demonstration?
```

Promotion criteria:

- shuffled event labels must not help;
- task-only or time-index-only controls must not explain the main signal;
- visual+proprio+future-motion should beat proprio+future-motion for phase
  prediction if visual grounding is genuinely useful;
- event-aware scorer must improve action MSE or oracle rank before it becomes
  part of the method.
