# GeoMoCo-WM Design Gates: cVAE, Oracle Motion, and Action Decoder

- Date: 2026-06-04
- Project: `Geomoco-WM`
- Source context: discussion after reviewing old GeoMoCo worklog, DINO grounding results, ZipMotion, AMPLIFY, and Fast-WAM/Wan design patterns.

## User Questions

1. Prior GeoMoCo experiments showed that adding DINO visual grounding did not clearly improve downstream closed-loop success. Does a GeoMoCo-cVAE design have a real chance to improve performance?
2. Could the old tasks have been too narrow or saturated to reveal GeoMoCo's benefit?
3. Given the old evidence, should the new GeoMoCo-WM design be modified?
4. What does `oracle future motion` mean?
5. What does this gate mean: `GT future motion -> action decoder` must clearly outperform `direct BC`; otherwise the action-decoder route is not worth continuing?

## Short Answer

The design should be modified from "build DINO + GeoMoCo-cVAE + action decoder directly" to a gated research program.

GeoMoCo should first be tested as an `SE(3)`-faithful future-motion state factor. A cVAE is only valuable if the bottleneck is truly multimodal future-motion uncertainty, phase/progress ambiguity, or long-horizon geometric planning. If even privileged ground-truth future motion cannot help a shared action decoder outperform direct behavior cloning, then the downstream control problem is not waiting for a better motion prior; in that case a cVAE connected to an action decoder will probably not improve success.

## First-Principles Reasoning

A world-motion representation helps control only if three conditions hold:

1. The representation contains task-relevant information that the baseline policy does not already extract.
2. The downstream decoder can turn that representation into executable actions.
3. The benchmark has enough headroom and diversity for the extra information to matter.

Old GeoMoCo evidence suggests that GeoMoCo's strongest signal was predictive or phase-factor value, not robust closed-loop policy improvement. DINO grounding improved candidate recall in some cases, but did not reliably solve progress precision, reranking, or execution. Therefore visual grounding and cVAE sampling should not be treated as guaranteed upgrades.

The new project should separate four questions:

1. Does visual grounding improve future motion/progress prediction?
2. Does a stochastic future-motion model improve coverage over deterministic AE or direct residual prediction?
3. Does future motion, if provided perfectly, actually help action decoding?
4. Does predicted or sampled future motion help closed-loop control after the above gates pass?

## Meaning Of Oracle Future Motion

`Oracle future motion` means privileged ground-truth future motion from demonstrations or evaluation traces. It is not available to a real deployed policy. It is used as an upper-bound diagnostic.

Example:

- At time `t`, the real policy only sees observation/history/task.
- The oracle condition is allowed to read the actual future EEF/object/motion trajectory from `t+1 ... t+H`.
- That future motion is then given to the same action decoder.

If this privileged future signal improves performance, then future-motion representation may be a useful interface. If it does not, the failure is probably in action execution, task supervision, benchmark saturation, or the fact that direct BC already captures the needed signal.

## Meaning Of The GT-Motion Action-Decoder Gate

Gate statement:

> Only if `GT future motion -> action decoder` clearly outperforms `direct BC` is the action-decoder route worth continuing.

Interpretation:

- `direct BC`: observation/history/task -> action chunk.
- `GT future motion -> action decoder`: observation/history/task plus privileged ground-truth future motion -> action chunk.
- The action decoder should be shared or controlled so the comparison isolates the value of future motion, not decoder capacity.

This gate asks:

> If I give the decoder the answer key for future motion, can it act better than a normal policy?

If the answer is no, then the bottleneck is not motion-prior quality. A learned cVAE will produce noisier, less accurate future motion than the oracle, so it is unlikely to beat direct BC.

If the answer is yes, then the future-motion interface has causal value, and it becomes meaningful to ask whether GeoMoCo-AE or GeoMoCo-cVAE can approximate that oracle benefit.

## Updated Design

Recommended first route:

1. Mine or select tasks with visible headroom:
   - avoid tasks where direct BC is already saturated;
   - avoid tasks where all policies collapse for unrelated reasons;
   - prefer long-horizon, phase-sensitive, multimodal, contact-rich, or object-relative manipulation.
2. Run predictive grounding gates:
   - state/history only;
   - DINO only;
   - state + DINO;
   - GeoMoCo deterministic AE;
   - GeoMoCo-cVAE;
   - random/shuffled latent controls;
   - oracle future-motion upper bound.
3. Test future-motion coverage:
   - cVAE should improve min-of-K future EEF/progress/contact coverage over AE and direct residual prediction;
   - do not accept reconstruction-only gains as enough evidence.
4. Run the oracle action-decoder gate:
   - `GT future motion -> action decoder` must beat `direct BC`;
   - otherwise stop or redesign the action interface.
5. Promote to learned action decoding only after the oracle gate passes:
   - compare AE latent, cVAE samples, best-of-K samples, and direct BC under a shared decoder setting.
6. Run closed-loop LIBERO only after predictive and offline gates show non-degenerate value.

## Position Relative To ZipMotion And AMPLIFY

ZipMotion is useful as a long-horizon continuous motion-latent reference. It suggests that a compressed future-motion latent can act as a planning interface before low-level execution.

AMPLIFY is useful as an actionless-video-to-action reference. Its most relevant piece is not necessarily the discrete FSQ tokenizer itself, but the separation between motion prior and inverse-dynamics/action decoder.

For GeoMoCo-WM, the most interesting hybrid idea remains:

> ZipMotion-style long-horizon motion latent + AMPLIFY-style inverse dynamics/action decoder, but with GeoMoCo's `SE(3)`-faithful geometric motion factor as the motion state.

However, this hybrid should be pursued only after the gates show that future motion has executable value.

## Decision

Do not make GeoMoCo-cVAE the default main claim yet.

Make the paper route:

> GeoMoCo as a geometry-faithful future-motion state factor, validated by gated predictive, oracle, and controlled action-decoder tests.

Make cVAE a promoted module only if it improves multimodal future-motion coverage and moves toward the oracle upper bound.

Make the action decoder a diagnostic/executor first, not a new full VLA policy. A complete action policy is worthwhile only if the oracle future-motion gate passes and the learned motion representation closes part of that gap.

## Follow-Up Experiments

1. Build the dataset contract with future EEF/object-relative motion and action chunks.
2. Select task suites by headroom and phase sensitivity before training large models.
3. Implement oracle future-motion baselines early.
4. Run direct BC versus GT future-motion action decoder before investing in large cVAE training.
5. Treat DINO grounding as a hypothesis to test, not as a guaranteed performance source.
