# Phase / Progress / Composition Supervision

- Date: 2026-06-06
- Project: `Geomoco-WM`
- Context: discussion after OASIS, ZipMotion / ZipMo, AMPLIFY, Fast-WAM, old GeoMoCo progress audits, and the first oracle future-motion action-decoder smoke.

## User Question

For `phase`, `progress`, and `composition`, are there existing works that define quantitative labels or metrics? Can those definitions be borrowed as supervision for GeoMoCo-WM, or should the project keep using the old GeoMoCo `u_t` definition and heuristic targets?

## Short Answer

There are useful quantitative traditions, but none should replace the old GeoMoCo `u_t` blindly.

Use old GeoMoCo `u_t` as the main low-cost geometric anchor, define it narrowly as normalized geometric motion progress / motion phase, and add auxiliary supervision plus diagnostics from temporal alignment, skill segmentation, object-state progress, visual change, and `SE(3)` composition metrics.

The main claim should not be that GeoMoCo-WM has a universal task-progress label. The stronger claim is:

```text
GeoMoCo-WM learns a compositional motion state factor.
The geometric progress coordinate is a scaffold for segment construction,
temporal probing, and composition evaluation.
```

## Definitions

`phase`, `progress`, and `composition` should stay separated.

| Concept | First-principles meaning | Suitable supervision or metric |
| --- | --- | --- |
| `phase` | where the current state lies inside a reusable motion or skill segment | phase bins, temporal ordering, temporal alignment, boundary F1 |
| `progress` | how far the rollout has advanced under a chosen coordinate or task variable | `u_t` RMSE / MAE, monotonicity violation, time-to-go, object-progress RMSE |
| `composition` | whether two motion segments combine into a valid longer motion | `SE(3)` geodesic closure error, latent closure error, hard-negative pair accuracy |

Do not force all three into one scalar. A scalar `u_t` can organize motion, but composition is an algebraic relation and semantic progress is task-dependent.

## Existing Work Families To Borrow From

1. Movement primitives such as DMP / ProMP

   These commonly use phase variables, canonical systems, or normalized time to align demonstrations with different speeds. They are closest in spirit to old GeoMoCo `u_t`: cheap, continuous, and useful for motion organization, but not semantic task completion.

2. Temporal alignment and video imitation

   Methods such as temporal cycle consistency, time-contrastive learning, soft-DTW alignment, and ordering losses provide weak or self-supervised phase labels from demonstration ordering. They are useful auxiliary objectives when there are multiple demos of the same task.

3. Skill segmentation and options

   These define phase through segment boundaries, skill IDs, or option termination. Useful metrics include boundary F1, segment purity, transition accuracy, and phase-bin classification. They are more semantic, but usually require labels or a separate discovery stage.

4. Task-progress and value-style signals

   Examples include success probability, time-to-go, distance-to-goal, drawer joint position, button state, or object displacement. These are stronger task-progress labels but are task-specific and often require privileged simulator state.

5. Geometric rigid-motion and composition metrics

   For an `SE(3)` segment convention such as `T_i_j`, composition can be evaluated by a Lie-log geodesic closure error:

   ```text
   T_i_k ~= T_j_k * T_i_j
   error = || Log( T_i_k^{-1} (T_j_k * T_i_j) ) ||_W
   ```

   In latent space, evaluate whether `compose(z_A, z_B)` recovers `z_AB`, and whether the positive `AB` candidate beats a near but wrong negative.

## Old GeoMoCo `u_t`

Old GeoMoCo used normalized cumulative EEF geometric motion:

```text
step_length = ||p_{s+1} - p_s||_2 + alpha * d_R(R_s, R_{s+1})
u_t = cumulative_step_length(0 -> t) / cumulative_step_length(0 -> T)
```

This target is useful because it:

- normalizes away frame-rate and execution-speed differences;
- supports segment sampling and triplet construction;
- provides a stable low-cost motion-phase coordinate;
- preserves continuity with old GeoMoCo evidence.

Its limits must stay explicit:

- it is EEF-centric;
- it does not use object pose, contact, gripper events, visual change, or success predicates;
- it is monotonic whenever the robot moves, even during retry, detour, or irrelevant motion;
- it is not a universal task-progress label.

## Recommended Target Stack

Use a layered target stack rather than choosing one label.

```text
Primary anchor:
  u_geom = old GeoMoCo normalized EEF geometric progress

Cheap auxiliary labels:
  phase_bin = discretized u_geom
  temporal_rank = later window should be ahead of earlier window
  gripper_event = open / close / contact-adjacent phase landmark when available

Weak grounding labels:
  u_vis = cumulative frozen visual/depth feature change
  future_visual_feature = short-horizon DINO/depth feature prediction

Privileged diagnostic labels:
  u_object = normalized task-relevant object or joint progress if clean
  contact / success predicate = upper-bound or sanity probe only

Core composition labels:
  SE(3) segment closure
  latent composition closure
  hard-negative pair ranking
```

## Metrics

Report geometry, phase, semantics, and control value separately.

| Metric | Purpose |
| --- | --- |
| `u_geom` RMSE / MAE | does the model retain normalized motion phase? |
| monotonicity violation rate | does learned progress move forward consistently? |
| phase-bin accuracy / F1 | are coarse stages recoverable? |
| temporal rank accuracy | does the representation order earlier/later windows correctly? |
| top-K future-phase coverage | does retrieval or sampling include plausible future-phase candidates? |
| hard-negative pair accuracy / margin | does the model prefer correct composition over near wrong windows? |
| `SE(3)` geodesic composition error | is the explicit motion composition geometrically valid? |
| latent closure error | does the representation preserve compositional algebra? |
| future state/progress RMSE | does the latent act as a predictive world-motion factor? |
| oracle / learned motion action-decoder gap | does motion information actually improve action generation? |

## Decision

Use the old `u_t` as the anchor, not because it is perfect, but because it is the cleanest low-cost coordinate for the current claim.

Do not promote new heuristics to main supervision until they pass ablations against `u_t`. Treat them as auxiliary probes or regularizers first.

The first experiment matrix should compare:

```text
u_geom only
u_geom + temporal rank
u_geom + gripper/contact event
u_geom + visual feature change
u_geom + object-progress diagnostic if available
```

The key promotion condition is not just lower `u_t` error. A target is useful only if it improves at least one downstream mechanism gate:

```text
future state/progress prediction
future-motion coverage
composition hard-negative ranking
oracle-to-learned action-decoder gap
closed-loop success after predictive gates pass
```

## Design Implication For GeoMoCo-WM

GeoMoCo-WM should continue focusing on:

- phase/progress as motion-state organization;
- composition as the structural novelty;
- visual/depth grounding as context alignment;
- learned action decoding as an executor, not a hardcoded `SE(3)` trajectory-to-action converter.

This preserves the OASIS lesson: explicit motion intermediates help when they are grounded and decoded by a learned policy head, but hard-decoding EEF motion into action is brittle.

## Next Steps

1. Add `u_geom` computation to the LIBERO window export or the first target-building stage.
2. Add phase-bin and temporal-rank labels without changing the model architecture.
3. Add composition triplet diagnostics before claiming learned phase semantics.
4. Keep object-progress labels diagnostic-only until a dataset provides clean task-relevant object state.
5. Use action-decoder and predictive gates to decide whether any progress heuristic is genuinely useful.
