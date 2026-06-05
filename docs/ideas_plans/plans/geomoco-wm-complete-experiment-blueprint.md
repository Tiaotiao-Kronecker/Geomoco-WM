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
