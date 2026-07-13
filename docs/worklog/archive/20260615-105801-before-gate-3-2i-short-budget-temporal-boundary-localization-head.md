# Active Worklog

- Last updated: 2026-06-15
- Repository: `/home/user/projects/Geomoco-WM`

## Current Objective
- Build `Geomoco-WM` as a visual-grounded GeoMoCo world-motion project:
  DINO visual grounding, GeoMoCo-AE / GeoMoCo-cVAE future-motion priors, and a
  controlled action decoder for predictive and closed-loop validation.

## Current Execution Slice
- Cross-paper design synthesis archived for how Fast-WAM, ZipMo, AMPLIFY,
  OASIS, SDP, GuidedVLA, MVP, and structured `SE(3)` world-model work should
  shape the GeoMoCo-WM mainline.
- Gate 1.6 two-file four-suite oracle action-decoder geodesic replacement
  completed.
- Gate 2 deterministic context-only future-motion prior completed as a
  diagnostic baseline.
- Gate 2.1 suite/task-conditioned future-motion prior completed as a stronger
  diagnostic baseline; it improves over Gate 2 but still does not beat direct
  context as a policy interface.
- Gate 2.2a DINOv2 global visual future-motion prior completed as the first
  learned prior that beats direct context through the frozen action-decoder
  interface.
- Gate 2.2b patch-pooled DINOv2 cross-attention future-motion prior completed
  and improves over Gate 2.2a.
- Gate 2.2c visual controls completed: shuffled DINO features fail while both
  real single-camera branches beat direct context, so the Gate 2.2 visual gain
  is attributable to aligned visual grounding.
- Gate 2.3a action-aware visual future-motion prior completed; a small frozen
  action-decoder auxiliary loss improves the best learned action MSE from
  `0.049547` to `0.043174`.
- Gate 2.3b action-aware lambda sweep completed; `lambda_action=0.030`
  becomes the action-value-prioritized deterministic default with mean action
  MSE `0.042090`, while `lambda_action=0.010` remains the balanced geometry
  reference.
- Gate 2.4a stepwise multi-query visual predictor completed; it improves
  future-motion translation geometry relative to the `0.030` single-query
  branch but does not improve action MSE, so it is not promoted as the default.
- Gate 2.4b first visual-conditioned cVAE future-motion prior completed; its
  prior mean slightly improves mean action MSE to `0.041579`, but KL is near
  zero so stochastic/multimodal behavior is not yet established.
- Gate 2.4c cVAE stochasticity calibration completed; free-bits raises raw KL
  from `0.000740` to `0.442097`, improves best-of-K action MSE from
  `0.039526` to `0.036894`, and slightly improves prior-mean action MSE to
  `0.040931`, but naive random samples remain worse than the prior mean.
- Gate 2.4d sample scorer/readout plan drafted; published-work precedents
  include BCQ-style candidate generation plus Q selection, IBC-style energy
  scoring, Trajectory Transformer beam/search readout, Diffuser/QGPO
  reward-energy guidance, and Visual MPC/PETS cost-based readout.
- Gate 2.4d is now explicitly framed as the minimal world-model planning layer:
  cVAE samples multiple future EEF motion rollouts, ScoreNet reads out one or
  aggregates them, and the action decoder remains deterministic for clean
  attribution before any multimodal action head is added.
- Gate 2.4d first lightweight ScoreNet readout completed; ScoreNet argmax
  improves mean action MSE from prior mean `0.040931` to `0.040201`, but closes
  only `18.09%` of the oracle readout gap, so gripper/contact/executability
  scoring remains the next bottleneck.
- Gate 2.4e structured SE(3)+gripper scorer completed; naive structured target
  replacement improves over prior mean but regresses from the Gate 2.4d flat
  action-MSE scorer (`0.040424` vs `0.040201`), so the next readout branch
  should use hard negatives or richer executability/contact proxies.
- Gate 2.4f structured oracle readout evaluation completed; SE(3) and
  SE(3)+gripper scorers improve the oracle ranks they target, but still lose
  on deployable action MSE, confirming that metric-target replacement is only
  diagnostic and not the next mainline path.
- Gate 2.4g hard-negative readout completed; naive SE(3)+gripper hard-negative
  auxiliary training regresses from the Gate 2.4d flat ScoreNet baseline, so
  the next readout work should build explicit gripper-transition/contact or
  executability proxies instead of selecting negatives from structured scores
  alone.
- Gate 2.4h-a gripper/event label audit completed; command-state labels are
  shortcut-prone, while transition labels isolate close/open phase-boundary
  events without the detected step-0 shortcut.
- Gate 2.4h-b visual phase/event probe completed; real DINO visual grounding
  strongly predicts transition labels and shuffled visual collapses, so visual
  context contains aligned phase/composition information.
- Gate 2.4h-c cVAE sample event-alignment analysis completed; event oracle
  best-of-K improves event accuracy from `0.836666` to `0.849049`, but
  transition step-within-1 coverage reaches only `0.219359` and flat ScoreNet
  reaches `0.175082`, so current samples/readout only weakly capture
  close/open timing.
- Gate 2.4h-d minimal event-aware readout completed; weak event ranking
  auxiliary (`event_target_weight=0.1`) slightly regresses action MSE
  (`0.040201 -> 0.040255`) and does not improve event metrics, so it is not
  promoted.
- Gate 2.4i event-fidelity interface audit completed; adding oracle future
  gripper to oracle future EEF reduces action MSE from `0.031474` to
  `0.004202`, gripper MSE from `0.184683` to `0.000241`, and makes transition
  event metrics essentially perfect, proving the EEF-only interface is missing
  the gripper/event channel.
- Gate 2.5a visual future-gripper/event predictor completed; real DINO
  patchpool visual grounding predicts future gripper/event timing better than
  task/proprio and shuffled visual controls, and `GT future EEF + predicted
  visual gripper` improves the bridge action MSE from the Gate 2.4i EEF-only
  oracle `0.031474` to `0.028987`.
- Gate 2.5b modular predicted EEF + predicted gripper bridge completed; real
  visual remains better than task/proprio and shuffled controls, but the
  modular bridge action MSE `0.050333` does not beat the previous best
  EEF-only learned prior `0.042090`, so separate EEF and gripper predictors are
  not promoted as the final joint interface.
- Gate 2.5b-joint deterministic `future_delta_gripper` predictor completed;
  increasing action-aware loss to `0.300` reaches action MSE `0.040688`,
  beating the modular bridge `0.050333` and the previous EEF-only learned prior
  `0.042090`; real visual also beats task/proprio and shuffled controls on
  action, SE(3), gripper, and transition-event metrics.
- Gate 2.5c joint GeoMoCo-cVAE completed as a mixed-positive result; real
  visual cVAE prior mean action MSE `0.043816` does not beat deterministic
  joint `0.040688`, but best-of-K action MSE reaches `0.022139` and real visual
  beats shuffled visual, so the joint cVAE sample space is useful but needs a
  deployable readout.
- Gate 2.5d joint cVAE sample readout completed as a weak-positive but not
  promoted result; real visual scorer argmax slightly improves prior-mean action
  MSE from `0.043816` to `0.043414`, while shuffled visual regresses from
  `0.068816` to `0.070023`. The readout is real but shallow: it still does not
  beat the deterministic joint baseline `0.040688`, and the oracle gap remains
  wide. Event readout improves modestly on real visual and more strongly on
  shuffled, but not enough to promote the current flat scorer.
- Gate 2.5e event-aware joint readout pilot completed as a useful negative:
  event-aware targets and event hard negatives improve transition-event
  alignment on seed 7, but action MSE regresses from the Gate 2.5d flat scorer
  (`0.045230`) to `0.045455`, `0.045710`, and `0.047043`. Do not expand this
  event-weight sweep until the sample/action interface or temporal readout is
  improved.
- 2026-06-10 cVAE loss/eval clarification archived:
  `docs/agent_qa/2026-06-10-cvae-loss-kl-freebits-beta-and-eval-contract.md`.
  Keep the distinction explicit: cVAE loss creates future candidates, ScoreNet
  loss chooses among frozen candidates, and promotion still uses the fixed
  downstream action-eval contract with event metrics as diagnostics.
- Gate 2.6a temporal action-regret readout completed as a weak negative:
  `TemporalSampleScoreNet` improves selected oracle rank slightly
  (`5.637193 -> 5.551251`) and transition step@1 (`0.264299 -> 0.289056`),
  but mean action MSE regresses from the Gate 2.5d flat scorer `0.043414` to
  `0.043636`. Temporal capacity alone is not enough; the next readout should
  explicitly predict calibrated downstream action regret or compare candidates
  set-wise.
- Gate 3.0a motion-prior-conditioned action head completed as the first
  positive downstream planner/head result: under a shared trained action-head
  protocol, real visual cVAE sample sets reach mean action MSE `0.036675`,
  beating context-only `0.037469`, real prior mean `0.037061`, and shuffled
  sample-set `0.042633`. Keep the interpretation scoped: this validates the
  multimodal motion-prior interface, but the numbers are not directly
  comparable to earlier frozen-action-decoder gates.
- Gate 3.0b K sweep and set-aggregator ablation completed: K=32 real samples
  reach mean action MSE `0.036565`, only `0.000110` better than K=16
  `0.036675`; `mean_pool`, `context_attention`, and `multi_query_attention`
  are effectively tied on real samples (`0.036691`, `0.036675`, `0.036949`).
  Keep `context_attention`, K=16 as default, with K=32 as an optional reference.
  Next bottleneck is sample-set diversity/action-head usage, not another small
  aggregator tweak.
- Gate 3.0c sample-set usage audit completed: original real K=16 sample sets
  reach mean action MSE `0.037061`, beating mean replacement `0.041893` and K=4
  subset `0.040585`; shuffled samples are more diverse (`pair_l2=1.385104` vs
  real `0.611503`) but worse (`0.042665`). The action head uses aligned
  sample-set diversity; generic unaligned diversity is harmful.
- 2026-06-11 sample-set usage diagnostics explanation archived:
  `docs/agent_qa/2026-06-11-sample-set-usage-diagnostics-explained.md`.
  Treat sample permutation, mean replacement, and batch mismatch as probes for
  whether the downstream action head is using aligned future-motion diversity.
- Gate 3.1 mode-structured / event-aware future-motion prior plan archived:
  `docs/experiments/plans/2026-06-11_gate3_1_mode_structured_event_aware_prior_plan.md`;
  decision summary:
  `docs/experiments/comparisons/2026-06-11_gate3_1_event_aware_prior_mainline_decision.md`.
  The next executable slice is Gate 3.1a event-mode target materialization and
  audit on the current two-file four-suite windows.
- Gate 3.1a event-mode target audit completed: close/open transition timing
  modes are measurable on the two-file four-suite slice, train/validation both
  contain all observed modes, and transition steps are not dominated by step 0.
  Mixed-transition modes are rare (`14` total windows), so keep them as
  diagnostics or merge them before class-balanced training.
- Gate 3.1b event-mode probe completed: real visual/proprio predicts stable
  event modes much better than task/proprio and shuffled visual controls
  (`macro-F1 0.448741` vs `0.306219` vs `0.090006`; transition F1 `0.599939`
  vs `0.423041` vs `0.191527`). This justifies adding event-mode conditioning
  to the cVAE.
- Gate 3.1c oracle-event conditioned cVAE completed as a strong upper-bound
  result: oracle event conditioning improves prior action MSE from `0.043816`
  to `0.018448` and best-of-K action MSE from `0.022139` to `0.014656`, while
  shuffled-event conditioning remains near the unconditional baseline. This
  validates event timing as a useful mode axis but is not deployable yet.
- Gate 3.1d predicted-event mixture completed as a positive sample-space
  result: predicted top-4 event modes reach event-mode coverage `0.981789` and
  best-of-K action MSE `0.015228`, close to the oracle-event best-of-K
  `0.014656`, but prior/readout action MSE remains `0.042992` and sample
  diversity is very large (`sample_pair_l2=2.704827`). The next step is to feed
  this predicted top-4 proposal set into the Gate 3 action head/planner.
- Gate 3.1e predicted-event mixture action head completed as a weak-positive
  control result: predicted top-2/top-4 action heads beat shuffled sample-set
  controls (`0.038052` / `0.038024` vs `0.042633`) but do not beat the simpler
  Gate 3.0 real unconditional sample-set baseline (`0.036675`). Do not promote
  naive anonymous sample-set aggregation over predicted event mixtures.
- Gate 3.1f event-aware sample consumption completed as the first positive
  predicted-event action-head result: adding per-sample event mode/rank/prob
  metadata lets predicted top-4 beat Gate 3.0 real sample-set action MSE
  (`0.034767` vs `0.036675`) and improves gripper MSE (`0.150052` vs
  `0.164061`). This validates event-structured sample consumption.
- Gate 3.1g event metadata ablation completed: full event/rank/prob metadata
  remains best (`0.034767` action MSE), rank/prob-only is useful (`0.036069`),
  event-only is weaker (`0.037108`), and shuffled event identity does not match
  the full aligned branch (`0.036228`). The gain needs both event identity and
  event-prior confidence.
- Gate 3.2a group stress audit completed: the event-aware top-4 interface
  reproduces its global result (`0.034773` action MSE), but transition windows
  are the dominant failure mode (`0.134087` transition MSE vs `0.022793`
  sustain MSE). The penalty is mostly gripper/open-close timing
  (`0.827336` transition gripper MSE), not generic SE(3) geometry.
- Gate 3.2b transition-weighted action head completed: transition loss weights
  reduce transition MSE (`0.134087 -> 0.125002 -> 0.122045` for baseline,
  weight 2, weight 4) and transition gripper MSE (`0.827336 -> 0.766794 ->
  0.742795`), but worsen overall MSE (`0.034773 -> 0.035986 -> 0.037981`)
  and sustain MSE (`0.022793 -> 0.025233 -> 0.027829`). This is
  mechanism-positive but not deployable as the default.
- Gate 3.2c auxiliary gripper action head completed as a negative ablation:
  an auxiliary gripper readout gives at most a tiny transition gain
  (`0.134087 -> 0.133936` for aux weight 0.3 aux-readout) while worsening
  overall MSE (`0.034767 -> 0.036388`) and not improving deployability.
  Simple parallel gripper regression is not enough; the next branch should use
  explicit event routing or transition-gated residuals.
- Gate 3.2d event-routed gripper residual completed as a useful negative:
  event-family route prediction is learnable (`route accuracy = 0.921017`),
  and routed output slightly reduces transition MSE (`0.138891 -> 0.138409`),
  but it worsens overall MSE (`0.035624 -> 0.035774`) and sustain MSE
  (`0.023131 -> 0.023355`). Window-level event routing is too coarse; the next
  branch should model step-wise gripper/event timing inside the action chunk.
- Gate 3.2e step-wise gripper command-timing head completed as a useful
  negative: per-step command-state prediction is easy (`step accuracy =
  0.947207`), but step-routed output worsens overall MSE (`0.035254 ->
  0.035397`) and transition MSE (`0.137397 -> 0.137707`). The issue is not
  per-step open/close command sign; the next branch should supervise the
  transition boundary itself with `close_step/open_step`.
- Gate 3.2f step-wise transition-boundary timing completed as a
  mechanism-positive but not deployable branch: boundary-start supervision
  slightly improves step-routed transition MSE (`0.137941 -> 0.137172`) and
  transition gripper MSE (`0.851940 -> 0.846643`), but worsens overall MSE
  (`0.035453 -> 0.035605`), gripper MSE (`0.154943 -> 0.156003`), and sustain
  MSE (`0.023101 -> 0.023363`). Boundary accuracy `0.986058` is mostly a
  sparsity artifact because positive boundary steps are only `0.013576` of
  labels. Do not promote it over the Gate 3.1f/Gate 3.1g reference.
- Gate 3.2g boundary-quality audit and positive-only repair completed as a
  weak mechanism-positive but deployable-negative result: the audit shows
  Gate 3.2f argmax boundary recall is only `0.007233`, while positive-only
  residual blending improves step transition MSE (`0.137172 -> 0.136703`) and
  transition gripper MSE (`0.846643 -> 0.833302`) but still worsens overall
  MSE versus the Gate 3.1f/Gate 3.1g reference (`0.035204` vs `0.034767`).
  Positive CE weighting raises boundary recall to `0.532779` but precision is
  only `0.094722` and transition MSE regresses to `0.142132`. Stop simple
  sparse-CE deterministic residual variants; the next deterministic slice
  should test oracle boundary masks or a direct transition-local gripper
  trajectory correction.
- Gate 3.2h oracle-boundary transition-local gripper correction completed as
  an oracle-positive but deployable-negative result: using true
  `close_step/open_step` masks improves overall MSE to `0.032018`,
  gripper MSE to `0.133072`, and transition MSE to `0.116469`, beating the
  Gate 3.1f/Gate 3.1g reference. Predicted masks fail to recover the gain:
  best predicted-mask overall MSE is about `0.035201`, with boundary AP
  `0.098873` and argmax recall `0.012026`. The local gripper-correction
  mechanism is valid only when timing is known; sparse boundary CE/predicted
  threshold gates remain non-deployable.
- 2026-06-10 motion-prior positioning discussion archived:
  `docs/agent_qa/2026-06-10-motion-prior-positioning-and-next-mainline.md`.
  The mainline should now treat GeoMoCo-WM as a visually grounded multimodal
  future-motion proposal prior, not as a module that must select one future by
  itself. The next primary experiment should be a downstream action head or
  planner that consumes the full set of sampled futures.
- 2026-06-10 PointWorld multimodal-rollout contrast archived:
  `docs/agent_qa/2026-06-10-pointworld-vs-geomoco-wm-multimodal-rollouts.md`.
  Treat PointWorld-style multiple rollouts as planner-side candidate action
  sequences rolled through an action-conditioned dense 3D world model, while
  GeoMoCo-WM's current multimodality is model-intrinsic
  `future_delta_ee + future_gripper/event` sampling. This keeps Gate 2.5d
  focused on sample readout/scorer rather than shifting prematurely to
  action-sequence MPC or a multimodal action head.
- 2026-06-10 related-work positioning advantage archived:
  `docs/agent_qa/2026-06-10-geomoco-wm-positioning-advantages-vs-related-work.md`.
  The project should not claim latent multi-rollout generation itself as novel;
  the sharper advantage is placing stochastic future rollout in an
  action-relevant geometric interface, `future_delta_ee + future_gripper/event`,
  where visual grounding, event timing, sample readout, and action decoding can
  be measured separately.
- Action Semantics Audit completed and the action metric contract is now
  upgraded from normalized SE(3)-aware split metrics to physical translation
  plus SO(3) geodesic rotation metrics.
- Next run target: Gate 3.2i decision slice. Either design a stronger temporal
  boundary-localization objective/head that can approach the Gate 3.2h oracle
  mask, or formally pivot to a richer temporal/flow action head after
  documenting that simple deterministic routing is exhausted.
- The exporter writes HDF5 episode references plus lightweight future
  EEF/action windows; DINO global features are now cached separately for
  Gate 2.2a.

## Latest Results
- 2026-06-09 Gate 2.5a visual future-gripper/event predictor completed:
  - plan:
    `docs/experiments/plans/2026-06-09_gate2_5a_visual_future_gripper_predictor_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-09_gate2_5a_visual_future_gripper_predictor.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-09_gate2_5a_future_gripper_visual_controls.md`;
  - code:
    `scripts/train_future_motion_predictor.py`,
    `scripts/evaluate_future_gripper_events.py`,
    `scripts/evaluate_predicted_gripper_action_bridge.py`,
    `tests/test_future_motion_predictor.py`;
  - mean future-gripper MSE:
    task/proprio `0.324415`, visual patchpool `0.172088`,
    shuffled visual `0.233726`;
  - mean transition accuracy:
    task/proprio `0.014386`, visual patchpool `0.634542`,
    shuffled visual `0.481249`;
  - mean bridge action MSE:
    task/proprio `0.050046`, visual patchpool `0.028987`,
    shuffled visual `0.037060`;
  - interpretation: visual grounding can predict a deployable gripper/event
    channel that partially repairs the EEF-only interface, but the branch still
    relies on GT future EEF for this diagnostic and remains far from the
    GT-gripper upper bound.
- 2026-06-09 Gate 2.5b modular predicted EEF + predicted gripper bridge
  completed:
  - plan:
    `docs/experiments/plans/2026-06-09_gate2_5b_predicted_joint_bridge_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-09_gate2_5b_predicted_joint_bridge.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-09_gate2_5b_joint_bridge_diagnostics.md`;
  - code:
    `scripts/evaluate_predicted_joint_action_bridge.py`;
  - mean action MSE:
    task/proprio `0.079633`, visual patchpool `0.050333`,
    shuffled visual `0.065466`;
  - decomposition:
    `GT EEF + predicted gripper = 0.028987`,
    `predicted EEF + GT gripper = 0.025443`,
    `predicted EEF + predicted gripper = 0.050333`;
  - interpretation: visual attribution remains positive, but separately
    trained EEF and gripper predictors compound noise in the joint action
    decoder; train a joint `future_delta_gripper` predictor next.
- 2026-06-09 Gate 2.5b-joint deterministic future-delta-gripper predictor
  completed:
  - plan:
    `docs/experiments/plans/2026-06-09_gate2_5b_joint_future_delta_gripper_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-09_gate2_5b_joint_future_delta_gripper_predictor.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-09_gate2_5b_joint_predictor_controls.md`;
  - code:
    `scripts/train_future_motion_predictor.py`,
    `scripts/evaluate_future_gripper_events.py`;
  - lambda result:
    visual joint `lambda=0.030` action MSE `0.049103`, visual joint
    `lambda=0.300` action MSE `0.040688`;
  - control result at `lambda=0.300`:
    task/proprio `0.084648`, visual patchpool `0.040688`, shuffled visual
    `0.063790`;
  - event result:
    transition accuracy task/proprio `0.009672`, visual patchpool `0.560270`,
    shuffled visual `0.330051`;
  - interpretation: joint EEF+gripper/event output is now the promoted
    deterministic representation and should be the target space for the next
    cVAE.
- 2026-06-09 Gate 2.5c joint GeoMoCo-cVAE completed:
  - plan:
    `docs/experiments/plans/2026-06-09_gate2_5c_joint_cvae_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-09_gate2_5c_joint_cvae_future_delta_gripper.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-09_gate2_5c_joint_cvae_vs_deterministic.md`;
  - code:
    `scripts/train_visual_cvae_future_motion.py`,
    `scripts/evaluate_visual_cvae_samples.py`,
    `scripts/evaluate_visual_cvae_gripper_events.py`;
  - best config:
    real visual, `motion_mode=future_delta_gripper`,
    `free_bits=0.02`, `beta_kl=0.001`, warmup 5,
    `prior_recon_weight=0.5`, `lambda_action=0.300`;
  - mean result:
    prior mean action MSE `0.043816`, deterministic joint baseline `0.040688`,
    best-of-K action MSE `0.022139`, shuffled prior mean action MSE `0.068816`;
  - event result:
    real prior mean transition accuracy `0.571094`, shuffled `0.223375`;
  - interpretation: raw prior mean is not promoted, but the stochastic joint
    sample set is strong and visually grounded; next branch should train a
    readout/scorer over joint cVAE samples.
- 2026-06-10 Gate 2.5e event-aware joint readout pilot completed:
  - plan:
    `docs/experiments/plans/2026-06-10_gate2_5e_event_aware_joint_sample_readout_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-10_gate2_5e_event_aware_joint_readout_pilot.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-10_gate2_5e_event_readout_vs_flat.md`;
  - code:
    `scripts/train_visual_cvae_sample_scorer.py`,
    `scripts/evaluate_visual_cvae_sample_scorer.py`,
    `tests/test_future_motion_predictor.py`;
  - seed-7 action MSE:
    Gate 2.5d flat `0.045230`, event `w=0.05` `0.045455`,
    event `w=0.10` `0.045710`, event hard-negative `0.047043`;
  - event result:
    event hard-negative improves event accuracy to `0.904429`,
    transition accuracy to `0.645995`, and step@1 to `0.284238`;
  - interpretation: the event signal is real but currently conflicts with
    action-value selection. Treat this as a stop signal for simple event-weight
    sweeps and move next toward temporal/action-regret readout or better joint
    sample training.
- 2026-06-10 Gate 2.6a temporal action-regret readout completed:
  - plan:
    `docs/experiments/plans/2026-06-10_gate2_6a_temporal_action_regret_readout_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-10_gate2_6a_temporal_action_regret_readout.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-10_gate2_6a_temporal_vs_flat_readout.md`;
  - code:
    `src/geomoco_wm/models/sample_readout.py`,
    `scripts/train_visual_cvae_sample_scorer.py`,
    `scripts/evaluate_visual_cvae_sample_scorer.py`,
    `scripts/evaluate_cvae_event_alignment.py`,
    `tests/test_future_motion_predictor.py`;
  - mean result:
    flat ScoreNet action MSE `0.043414`, temporal ScoreNet action MSE
    `0.043636`, temporal selected rank `5.551251`, temporal transition step@1
    `0.289056`;
  - interpretation: temporal sequence modeling changes the ranking signal but
    does not improve downstream action value. Do not promote temporal v1.
- 2026-06-10 Gate 3.0a motion-prior-conditioned action head completed:
  - plan:
    `docs/experiments/plans/2026-06-10_gate3_0a_motion_prior_conditioned_action_head_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-10_gate3_0a_motion_prior_conditioned_action_head.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-10_gate3_0a_action_head_prior_ablation.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_motion_prior_action_head.py`,
    `scripts/evaluate_motion_prior_action_head.py`,
    `tests/test_motion_prior_action_head.py`;
  - mean action MSE across seeds 7/17:
    context-only `0.037469`, real cVAE prior mean `0.037061`,
    real cVAE sample set K=16 `0.036675`, shuffled cVAE sample set K=16
    `0.042633`, GT future upper bound `0.004624`;
  - repeated-eval sample-set files:
    `outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/repeated_eval_5pass.json`,
    `outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed17/repeated_eval_5pass.json`,
    `outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed7/repeated_eval_5pass.json`,
    `outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed17/repeated_eval_5pass.json`;
  - interpretation: real visual motion-prior samples are useful to a
    downstream action head and clearly beat shuffled visual samples. The gain
    over prior mean is small, so the next step should improve set aggregation
    or K/capacity, not claim solved planning.
- 2026-06-11 Gate 3.0b K sweep and set aggregator ablation completed:
  - plan:
    `docs/experiments/plans/2026-06-10_gate3_0b_k_sweep_set_aggregator_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_0b_k_sweep_set_aggregator.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_0b_k_and_aggregator_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_motion_prior_action_head.py`,
    `scripts/evaluate_motion_prior_action_head.py`,
    `tests/test_motion_prior_action_head.py`;
  - K sweep mean action MSE:
    real K=4 `0.037166`, K=8 `0.037772`, K=16 `0.036675`,
    K=32 `0.036565`; shuffled K=4 `0.038868`, K=8 `0.040161`,
    K=16 `0.042633`, K=32 `0.041596`;
  - aggregator mean action MSE at K=16:
    real `context_attention=0.036675`, `mean_pool=0.036691`,
    `multi_query_attention=0.036949`; shuffled `context_attention=0.042633`,
    `mean_pool=0.042385`, `multi_query_attention=0.042333`;
  - interpretation: K=32 slightly improves real sample-set action prediction,
    but the effect is tiny and non-monotonic. Aggregator variants are close, and
    mean pooling nearly matches attention. Move next to a mode/diversity and
    action-head usage audit.
- 2026-06-11 Gate 3.0c sample-set usage audit completed:
  - plan:
    `docs/experiments/plans/2026-06-11_gate3_0c_sample_set_usage_audit_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_0c_sample_set_usage_audit.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_0c_usage_audit_summary.md`;
  - code:
    `scripts/audit_motion_prior_action_head_usage.py`,
    `tests/test_motion_prior_action_head_audit.py`;
  - mean result across seeds:
    real original `0.037061`, real mean-repeated `0.041893`,
    real K=4 subset `0.040585`, real batch-mismatch `0.317347`;
    shuffled original `0.042665`, shuffled mean-repeated `0.053407`,
    shuffled K=4 subset `0.048454`;
  - diversity result:
    real sample pair L2 `0.611503`, shuffled sample pair L2 `1.385104`;
    real best single-sample action MSE `0.019049`, shuffled `0.023408`;
  - interpretation: the action head is not merely reading a mean future.
    It uses aligned sample-set information. The next model-side step should make
    mode/event structure explicit instead of chasing generic sample diversity.
- 2026-06-11 Gate 3.1a event-mode target audit completed:
  - plan:
    `docs/experiments/plans/2026-06-11_gate3_1_mode_structured_event_aware_prior_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_1a_event_mode_target_audit.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_1a_event_mode_label_readiness.md`;
  - code:
    `src/geomoco_wm/data/event_modes.py`,
    `scripts/audit_event_modes.py`,
    `tests/test_event_modes.py`;
  - artifacts:
    `outputs/event_modes/gate3_1a_event_modes_2files.json`,
    `outputs/event_modes/gate3_1a_event_modes_2files.md`;
  - event-mode counts:
    `sustain_open::none=8938`, `sustain_close::none=5881`,
    `transition_close::{early,middle,late}=329/339/210`,
    `transition_open::{early,middle,late}=292/315/200`,
    `mixed_transition::{early,middle}=10/4`;
  - interpretation: close/open transition timing is measurable and balanced
    enough for Gate 3.1b event-mode probing. Mixed transitions are rare and
    should stay diagnostic or be merged before class-balanced training.
- 2026-06-11 Gate 3.1b event-mode probe completed:
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_1b_event_mode_probe.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_1b_event_mode_probe_summary.md`;
  - code:
    `scripts/train_event_mode_probe.py`,
    `tests/test_event_modes.py`;
  - artifacts:
    `outputs/event_mode_probe/gate3_1b_task_proprio_seed{7,17}/metrics.json`,
    `outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17}/metrics.json`,
    `outputs/event_mode_probe/gate3_1b_shuffled_visual_proprio_seed{7,17}/metrics.json`;
  - mean macro-F1:
    task/proprio `0.306219`, real visual/proprio `0.448741`,
    shuffled visual/proprio `0.090006`;
  - mean transition F1:
    task/proprio `0.423041`, real visual/proprio `0.599939`,
    shuffled visual/proprio `0.191527`;
  - interpretation: aligned visual features carry close/open event timing
    information, while shuffled visual collapses. Proceed to event-conditioned
    cVAE.
- 2026-06-11 Gate 3.1c oracle-event conditioned cVAE completed:
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_1c_oracle_event_conditioned_cvae.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_1c_oracle_event_cvae_summary.md`;
  - code:
    `src/geomoco_wm/data/event_conditioning.py`,
    `scripts/train_visual_cvae_future_motion.py`,
    `scripts/evaluate_visual_cvae_samples.py`,
    `scripts/train_motion_prior_action_head.py`,
    `tests/test_event_modes.py`;
  - artifacts:
    `outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/`,
    `outputs/visual_cvae_future_motion/gate3_1c_shuffled_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/`,
    `outputs/visual_cvae_samples/gate3_1c_oracle_event_joint_cvae_seed{7,17}_k16.json`,
    `outputs/visual_cvae_samples/gate3_1c_shuffled_event_joint_cvae_seed{7,17}_k16.json`;
  - mean action result:
    unconditional prior `0.043816`, shuffled-event prior `0.042406`,
    oracle-event prior `0.018448`; unconditional best-of-K `0.022139`,
    shuffled-event best-of-K `0.021296`, oracle-event best-of-K `0.014656`;
  - interpretation: correct event timing is a high-value mode axis for the
    joint future-motion prior. The deployable route must replace oracle event
    labels with predicted/top-M event modes.
- 2026-06-11 Gate 3.1d predicted-event mixture completed:
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_1d_predicted_event_mixture.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_1d_predicted_event_mixture_summary.md`;
  - code:
    `src/geomoco_wm/data/predicted_event_mixture.py`,
    `scripts/evaluate_predicted_event_cvae_mixture.py`,
    `tests/test_event_modes.py`;
  - artifacts:
    `outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed{7,17}_top1_k16.json`,
    `outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed{7,17}_top2_k16.json`,
    `outputs/visual_cvae_samples/gate3_1d_predicted_event_mixture_seed{7,17}_top4_k16.json`;
  - mean top-M result:
    top-1 coverage `0.813228`, best-of-K action MSE `0.042023`;
    top-2 coverage `0.879662`, best-of-K action MSE `0.027537`;
    top-4 coverage `0.981789`, best-of-K action MSE `0.015228`;
  - interpretation: predicted top-4 nearly recovers oracle-event best-of-K
    coverage, so good futures are present in the deployable sample set. The
    sample set is wide and noisy, so the next mainline is action-head/planner
    consumption rather than claiming the event mixture alone solves readout.
- 2026-06-11 Gate 3.1e predicted-event mixture action head completed:
  - formal record:
    `docs/experiments/runs/2026-06-11_gate3_1e_predicted_event_mixture_action_head.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-11_gate3_1e_action_head_summary.md`;
  - discussion archive:
    `docs/agent_qa/2026-06-11-gate31e-purpose-action-head-planner.md`;
  - code:
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_1e_pred_event_top2_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_1e_pred_event_top4_k16_seed{7,17}/`;
  - repeated-eval mean action MSE:
    Gate 3.0 real sample-set K=16 `0.036675`,
    Gate 3.0 shuffled sample-set K=16 `0.042633`,
    predicted top-2 `0.038052`, predicted top-4 `0.038024`;
  - interpretation: predicted event-mixture samples contain real signal because
    they beat shuffled controls, but anonymous sample-set aggregation does not
    exploit event structure enough to beat the unconditional real sample-set.
    Next step should expose event mode/rank/probability to the consumer.
- 2026-06-12 Gate 3.1f event-aware sample consumption completed:
  - formal record:
    `docs/experiments/runs/2026-06-12_gate3_1f_event_aware_sample_consumption.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-12_gate3_1f_event_aware_sample_consumption_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `tests/test_motion_prior_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_1f_eventaware_top2_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_1f_eventaware_top4_k16_seed{7,17}/`;
  - repeated-eval mean action MSE:
    Gate 3.0 real sample-set K=16 `0.036675`,
    Gate 3.1e anonymous top-4 `0.038024`,
    Gate 3.1f event-aware top-2 `0.036671`,
    Gate 3.1f event-aware top-4 `0.034767`;
  - interpretation: the predicted event mixture becomes useful when samples
    carry event identity/rank/probability. Top-4 is now the current best
    deployable action-head interface.
- 2026-06-12 Gate 3.1g event metadata ablation completed:
  - formal record:
    `docs/experiments/runs/2026-06-12_gate3_1g_event_metadata_ablation.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-12_gate3_1g_event_metadata_ablation_summary.md`;
  - code:
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `tests/test_motion_prior_action_head.py`;
  - repeated-eval mean action MSE:
    anonymous top-4 `0.038024`, event-only `0.037108`,
    rank/prob-only `0.036069`, shuffled-event/rank/prob `0.036228`,
    full event/rank/prob `0.034767`;
  - interpretation: rank/probability provides a useful confidence signal, but
    full aligned event identity plus confidence is needed for the strongest
    action-head result. The event-aware interface is mechanism-positive.
- 2026-06-12 Gate 3.2a group stress audit completed:
  - formal record:
    `docs/experiments/runs/2026-06-12_gate3_2a_group_stress_audit.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-12_gate3_2a_group_stress_summary.md`;
  - code:
    `scripts/audit_predicted_event_mixture_action_head_groups.py`,
    `tests/test_predicted_event_mixture_action_head_group_audit.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2a_group_stress_eventaware_top4_seed7/group_stress_3pass.json`,
    `outputs/motion_prior_action_head/gate3_2a_group_stress_eventaware_top4_seed17/group_stress_3pass.json`;
  - mean over seeds:
    overall action MSE `0.034773`, sustain action MSE `0.022793`,
    transition action MSE `0.134087`, transition-open action MSE `0.150220`,
    transition gripper MSE `0.827336`;
  - interpretation: Gate 3.1f/g remains globally stable, but the remaining
    error is concentrated in transition/open-close timing rather than SE(3)
    geometry. Gate 3.2b should target transition-balanced or transition-aware
    action-head training before any flow/diffusion action head.
- 2026-06-12 Gate 3.2b transition-weighted action head completed:
  - formal record:
    `docs/experiments/runs/2026-06-12_gate3_2b_transition_weighted_action_head.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-12_gate3_2b_transition_weighted_summary.md`;
  - code:
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `tests/test_motion_prior_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2b_transition_weight2_top4_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_2b_transition_weight4_top4_k16_seed{7,17}/`;
  - group stress means:
    baseline transition MSE `0.134087`, weight-2 `0.125002`,
    weight-4 `0.122045`; baseline transition gripper MSE `0.827336`,
    weight-2 `0.766794`, weight-4 `0.742795`;
  - trade-off:
    baseline overall MSE `0.034773`, weight-2 `0.035986`,
    weight-4 `0.037981`; baseline sustain MSE `0.022793`,
    weight-2 `0.025233`, weight-4 `0.027829`;
  - interpretation: transition timing is trainable, but scalar weighting
    moves error from transition windows into sustain/overall windows. Do not
    promote the weighted branch as the default; move to a structured
    transition-aware action head or auxiliary timing objective.
- 2026-06-12 Gate 3.2c auxiliary gripper action head completed:
  - formal record:
    `docs/experiments/runs/2026-06-12_gate3_2c_aux_gripper_action_head.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-12_gate3_2c_aux_gripper_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `scripts/audit_predicted_event_mixture_action_head_groups.py`,
    `tests/test_motion_prior_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2c_auxgripper_w0p3_top4_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_2c_auxgripper_w1p0_top4_k16_seed{7,17}/`;
  - mean result:
    baseline overall MSE `0.034767`, aux weight 0.3 aux-readout `0.036388`,
    aux weight 1.0 aux-readout `0.036599`; baseline transition MSE
    `0.134087`, aux weight 0.3 aux-readout `0.133936`, aux weight 1.0
    aux-readout `0.137685`;
  - interpretation: a separate gripper regression head does not repair
    transition timing in a deployable way. The next branch should use
    event-routed output or a transition-gated residual rather than another
    flat gripper regressor.
- 2026-06-12 Gate 3.2d event-routed gripper residual completed:
  - formal record:
    `docs/experiments/runs/2026-06-12_gate3_2d_event_routed_gripper_residual.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-12_gate3_2d_event_routed_residual_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `scripts/audit_predicted_event_mixture_action_head_groups.py`,
    `tests/test_motion_prior_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2d_event_routed_gripper_residual_top4_k16_seed{7,17}/`;
  - mean repeated-eval result:
    base overall MSE `0.035624`, routed overall MSE `0.035774`;
    base transition MSE `0.138891`, routed transition MSE `0.138409`;
    route accuracy `0.921017`;
  - interpretation: event family is readable, but window-level gripper
    residual routing is too coarse. It gives only a tiny transition gain while
    damaging sustain and overall metrics. The next branch should add
    step-wise gripper/event timing rather than more window-level routing.
- 2026-06-13 Gate 3.2e step-wise gripper command-timing head completed:
  - formal record:
    `docs/experiments/runs/2026-06-13_gate3_2e_step_gripper_timing_head.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-13_gate3_2e_step_gripper_timing_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `scripts/audit_predicted_event_mixture_action_head_groups.py`,
    `tests/test_motion_prior_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2e_step_gripper_timing_top4_k16_seed{7,17}/`;
  - mean repeated-eval result:
    base overall MSE `0.035254`, step-routed overall MSE `0.035397`;
    base transition MSE `0.137397`, step-routed transition MSE `0.137707`;
    gripper step command accuracy `0.947207`;
  - interpretation: command-state supervision is too easy and too coarse.
    The real bottleneck is transition-boundary timing, not whether each step's
    gripper command is open or close.
- 2026-06-13 Gate 3.2f step-wise transition-boundary timing head completed:
  - formal record:
    `docs/experiments/runs/2026-06-13_gate3_2f_boundary_timing_head.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-13_gate3_2f_boundary_timing_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `scripts/audit_predicted_event_mixture_action_head_groups.py`,
    `tests/test_motion_prior_action_head.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2f_boundary_timing_top4_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_2f_boundary_timing_smoke_seed7/`;
  - mean repeated-eval result:
    base overall MSE `0.035453`, step-routed overall MSE `0.035605`;
    base transition MSE `0.137941`, step-routed transition MSE `0.137172`;
    boundary-start accuracy `0.986058`, boundary positive fraction `0.013576`;
  - interpretation: boundary-start supervision gives a small targeted
    transition gain, but the target is very sparse and the step-routed branch
    regresses overall/gripper/sustain metrics. Keep this as mechanism evidence,
    not a deployable default.
- 2026-06-15 Gate 3.2g boundary-quality audit and positive-only repair
  completed:
  - formal record:
    `docs/experiments/runs/2026-06-15_gate3_2g_boundary_quality_and_positive_only_repair.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-15_gate3_2g_boundary_quality_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `scripts/audit_gripper_boundary_timing_head.py`,
    `tests/test_motion_prior_action_head.py`,
    `tests/test_gripper_boundary_timing_audit.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2g_boundary_positive_only_top4_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_2g_boundary_posw20_top4_k16_seed{7,17}/`;
  - mean repeated-eval result:
    3.2f step transition MSE `0.137172`;
    3.2g positive-only step transition MSE `0.136703`;
    3.2g positive-only step overall MSE `0.035204`;
    3.2g posw20 step transition MSE `0.142132`;
  - boundary audit:
    3.2f argmax recall `0.007233`, positive-only recall `0.001634`,
    posw20 recall `0.532779` but precision only `0.094722`;
  - interpretation: positive-only residual blending gives a small transition
    gain, but does not beat the deployable reference. Positive CE weighting
    fixes recall only by creating low-precision boundary fires that hurt action
    quality. Simple sparse-CE residual routing is exhausted.
- 2026-06-15 Gate 3.2h oracle-boundary transition-local gripper correction
  completed:
  - formal record:
    `docs/experiments/runs/2026-06-15_gate3_2h_oracle_boundary_transition_local_gripper_correction.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-15_gate3_2h_oracle_boundary_upper_bound_summary.md`;
  - code:
    `src/geomoco_wm/models/motion_prior_action_head.py`,
    `scripts/train_predicted_event_mixture_action_head.py`,
    `scripts/evaluate_predicted_event_mixture_action_head.py`,
    `scripts/audit_predicted_event_mixture_action_head_groups.py`,
    `tests/test_motion_prior_action_head.py`,
    `tests/test_predicted_event_mixture_action_head_group_audit.py`;
  - artifacts:
    `outputs/motion_prior_action_head/gate3_2h_oracle_boundary_top4_k16_seed{7,17}/`,
    `outputs/motion_prior_action_head/gate3_2h_oracle_boundary_smoke_seed7/`;
  - mean repeated-eval result:
    base overall MSE `0.035145`, soft step-routed MSE `0.034989`,
    oracle boundary-mask MSE `0.032018`; oracle transition MSE `0.116469`;
  - predicted-mask audit:
    best predicted-mask overall MSE is about `0.035201`, threshold sweeps from
    `0.05` to `0.50` either fire too often or too rarely, and boundary AP is
    `0.098873` with argmax recall `0.012026`;
  - interpretation: oracle boundary masks prove transition-local gripper
    correction is valuable and can beat the Gate 3.1f/Gate 3.1g reference, but
    the current predicted boundary head cannot recover the gain. Do not promote
    Gate 3.2h; choose between a stronger temporal boundary-localization head
    and a richer temporal/flow action decoder next.
- 2026-06-10 motion-prior positioning and next-mainline discussion archived:
  - note:
    `docs/agent_qa/2026-06-10-motion-prior-positioning-and-next-mainline.md`;
  - conclusion: providing a set of visually grounded multimodal future-motion
    priors whose samples contain good futures is itself a meaningful
    world-model/motion-prior contribution. Selection or aggregation can be
    delegated to a downstream action head or planner, provided controls show
    that real GeoMoCo-WM samples improve action prediction over direct context
    and shuffled-prior baselines.
- 2026-06-08 cross-paper GeoMoCo-WM design synthesis archived:
  - `docs/ideas_plans/plans/2026-06-08-cross-paper-lessons-for-geomoco-wm.md`;
  - core decision: GeoMoCo-WM should remain a visual-grounded stochastic
    world-motion prior with phase/progress/composition structure and controlled
    motion-to-action attribution, not collapse into an EEF trajectory policy,
    VLA finetuning recipe, video WAM, or strong action-head method;
  - immediate implication: Gate 2.4d sample scorer/readout is the next mainline
    world-model planning layer because calibrated cVAE samples contain useful
    futures but random sampling does not choose them reliably.
- Local project created at `/home/user/projects/Geomoco-WM`.
- GitHub remote configured as `git@github.com:Tiaotiao-Kronecker/Geomoco-WM.git`.
- Initial scaffold pushed to `origin/main` at commit `3718f19`.
- Core artifacts:
  - `README.md`
  - `pyproject.toml`
  - `src/geomoco_wm/models/geomoco_ae.py`
  - `src/geomoco_wm/models/geomoco_cvae.py`
  - `src/geomoco_wm/models/action_decoder.py`
  - `src/geomoco_wm/integrations/zipmo_adapter.py`
  - `src/geomoco_wm/integrations/amplify_adapter.py`
  - `experiments/geomoco_cvae/configs/minimal_libero.yaml`
  - `docs/ideas_plans/plans/geomoco-cvae-experiment-plan.md`
- Verification completed:
  - `python -m compileall src tests`
  - PyTorch smoke test for `GeoMoCoCVAE` and `ActionDecoder`
- `pytest` was not run because the active environment does not have `pytest`
  installed.
- 2026-06-04 route refinement:
  - old GeoMoCo evidence suggests EEF-centric latent is a phase/composition
    factor, not a full policy or world state;
  - a state-only cVAE would risk becoming only a multimodal version of old
    GeoMoCo;
  - current plan moves to DINO visual grounding plus GeoMoCo-cVAE future-motion
    prior;
  - Diffusion Policy / MeanFlow-style action heads are allowed as stronger
    shared decoders, but AMPLIFY is not required for the first version;
  - ZipMotion is deferred because DINO is a lighter first grounding front-end,
    while ZipMotion remains a stronger visual-motion extension/baseline.
- Added plan artifacts:
  - `docs/ideas_plans/plans/visual-grounded-geomoco-wm-plan.md`
  - `experiments/geomoco_cvae/configs/visual_grounded_libero.yaml`
- 2026-06-04 design-gate discussion archived:
  - `docs/agent_qa/2026-06-04-geomoco-wm-design-gates.md`
  - core decision: do not promote GeoMoCo-cVAE or the action-decoder route until
    predictive gates, future-motion coverage gates, and the oracle
    future-motion action-decoder gate show non-degenerate value.
- 2026-06-05 complete experiment blueprint archived:
  - `docs/ideas_plans/plans/geomoco-wm-complete-experiment-blueprint.md`
  - `docs/ideas_plans/html/geomoco-wm-complete-experiment-blueprint.html`
  - content covers experiment order, network architecture, dataset/window
    contract, DINO failure lessons, LIBERO-Long/LIBERO-10 promotion criteria,
    baseline matrix, metrics, and stop rules.
- 2026-06-05 Gate 0 HDF5 inspection slice written for review:
  - `src/geomoco_wm/data/libero_hdf5_inspect.py`
  - `scripts/inspect_libero_hdf5.py`
  - `tests/test_libero_hdf5_inspect.py`
  - `docs/ideas_plans/plans/gate0-libero-hdf5-inspection-runbook.md`
- Gate 0 verification completed on synthetic HDF5 only:
  - `python -m compileall src scripts tests`
  - `python -m unittest discover -s tests -p 'test_libero_hdf5_inspect.py'`
  - result: 4 tests passed.
- Gate 0 smoke inspection executed on real local LIBERO data:
  - command: `python scripts/inspect_libero_hdf5.py --input-path /home/user/dataset/libero_official/libero_goal --suite-name libero_goal --max-files 1 --max-demos-per-file 2 --output-json outputs/gate0/libero_goal_hdf5_inspection_smoke.json --output-md outputs/gate0/libero_goal_hdf5_inspection_smoke.md`
  - result: 1 file, 2 demos, 276 frames;
  - readiness: visual grounding, dual camera, EEF motion, action chunks, and
    proprio context all supported;
  - warning: object-state teacher fields unavailable, keep diagnostic-only.
- Gate 0 full `libero_goal` metadata inspection executed:
  - report: `outputs/gate0/libero_goal_hdf5_inspection.md`
  - JSON: `outputs/gate0/libero_goal_hdf5_inspection.json`
  - result: 10 files, 500 demos, 63,728 frames;
  - demo length range: 75 to 347, mean 127.456;
  - all required fields present, all sequence lengths aligned, all actions 7D,
    all EEF states 6D, all gripper states 2D, all joint states 7D;
  - `supports_gate0_dataset_export: true`;
  - `supports_object_state_teacher: false`.
- 2026-06-05 Gate 1 first exporter implemented:
  - `src/geomoco_wm/data/libero_hdf5_export.py`
  - `scripts/export_libero_windows.py`
  - `tests/test_libero_hdf5_export.py`
  - `docs/ideas_plans/plans/gate1-libero-window-export-runbook.md`
- Gate 1 verification completed:
  - `python -m compileall src scripts tests`
  - `python -m unittest discover -s tests -p 'test_libero_hdf5_*.py'`
  - result: 7 tests passed.
- Tiny real-data exporter smoke completed:
  - command: `python scripts/export_libero_windows.py --input-path /home/user/dataset/libero_official/libero_goal --suite-name libero_goal --output-dir outputs/libero_windows/libero_goal_smoke --context-len 2 --horizon 8 --stride 4 --max-files 1 --max-demos-per-file 1 --max-windows 3`
  - output: `outputs/libero_windows/libero_goal_smoke/`
  - result: 1 episode record, 3 window records, 138 source frames;
  - expected warning: `max_windows` reached.
- Experiment-facing oracle action-decoder path added:
  - `src/geomoco_wm/data/window_dataset.py`
  - `scripts/train_oracle_action_decoder.py`
  - dry-run result on smoke windows: 3 windows, context dim 15, motion dim 48,
    action dim 7, horizon 8;
  - 1-epoch CPU smoke completed and wrote
    `outputs/oracle_action_decoder/libero_goal_smoke/metrics.json`;
  - this is only a loop smoke, not a meaningful performance result.
- Motion-to-action and decoder discussion archived:
  - `docs/agent_qa/2026-06-05-motion-to-action-and-decoder-plan.md`
  - decision: use MLP as the first attribution-clean diagnostic, add stronger
    temporal/diffusion/flow decoders only after the oracle future-motion gate
    has signal.
- 10-demo drawer export and oracle-vs-direct MLP smoke completed:
  - export: `outputs/libero_windows/libero_goal_task0_10demo/`
  - result: 10 demos, 307 windows, 1,383 frames;
  - direct-context MLP final val MSE / MAE: `0.035896` / `0.113958`;
  - GT future EEF delta MLP final val MSE / MAE: `0.017163` / `0.088143`;
  - relative validation improvement: 52.19% MSE reduction, 22.65% MAE
    reduction;
  - report: `docs/agent_qa/2026-06-05-oracle-action-decoder-10demo-smoke.md`.
- 2026-06-06 local LIBERO data expanded from `libero_goal` to the standard
  four-suite set under `/home/user/dataset/libero_official`:
  - `libero_spatial`: 10 HDF5 files, 5.9G, `hdf5_ok=10`;
  - `libero_object`: 10 HDF5 files, 7.0G, `hdf5_ok=10`;
  - `libero_goal`: 10 HDF5 files, 6.0G, `hdf5_ok=10`;
  - `libero_10`: 10 HDF5 files, 13G, `hdf5_ok=10`.
- 2026-06-06 four-suite batch exporter and oracle input support added:
  - `src/geomoco_wm/data/libero_hdf5_export.py` now has
    `export_libero_hdf5_suite_collection`;
  - `scripts/export_libero_windows.py` now supports `--all-libero-suites`,
    `--suite-names`, per-suite caps, per-suite outputs, and combined
    `episodes.jsonl` / `windows.jsonl`;
  - `src/geomoco_wm/data/window_dataset.py` and
    `scripts/train_oracle_action_decoder.py` now support one or more
    `windows.jsonl` inputs and record suite/task counts;
  - oracle train/val split can now use `--split-by episode` for real
    comparisons.
- Four-suite smoke completed:
  - export:
    `outputs/libero_windows/libero_all_suites_smoke/`;
  - result: 4 suites, 4 files, 4 episodes, 8 windows, 656 frames;
  - dry-run future-delta spec: 8 windows, context dim 15, motion dim 48,
    action dim 7, horizon 8, suite counts 2 each;
  - dry-run direct-context spec: same dataset, motion dim 0;
  - 1-epoch loop smokes wrote:
    `outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta_1epoch/`
    and
    `outputs/oracle_action_decoder/libero_all_suites_smoke_direct_context_1epoch/`.
- Episode-level split and four-suite slice discussion archived:
  - `docs/agent_qa/2026-06-06-episode-split-and-four-suite-slice.md`;
  - decision: promote direct-context vs GT-future-motion from single-task smoke
    to four-suite episode-level comparison before cVAE/DINO claims.
- Four-suite test dataset usage archived:
  - `docs/agent_qa/2026-06-06-four-suite-test-dataset-usage.md`;
  - local dataset root: `/home/user/dataset/libero_official`;
  - standard suites present and validated: `libero_spatial`, `libero_object`,
    `libero_goal`, and `libero_10`, each with 10 readable HDF5 files;
  - dataset-use policy: smoke checks plumbing only, small formal slice uses
    one task file per suite with all demos, and full four-suite export waits
    until JSONL size and training speed are acceptable.
- Four-suite small formal slice exported:
  - output: `outputs/libero_windows/libero_all_suites_1file_all_demos_h8/`;
  - result: 4 suites, 4 HDF5 task files, 200 demos, 7,921 windows, 33,201
    frames;
  - no dropped short episodes and no exporter warnings;
  - combined `windows.jsonl` size: 33M.
- Four-suite oracle action-decoder comparison completed:
  - report:
    `docs/agent_qa/2026-06-06-four-suite-oracle-action-results.md`;
  - split: episode-level;
  - model: MLP `ActionDecoder`, hidden dims `256,256`, 20 epochs,
    batch size 64;
  - seed 7 direct-context val MSE / MAE: `0.081479` / `0.160300`;
  - seed 7 GT-future-motion val MSE / MAE: `0.035064` / `0.085608`;
  - seed 7 relative reduction: 56.97% MSE, 46.60% MAE;
  - seed 17 direct-context val MSE / MAE: `0.065460` / `0.149165`;
  - seed 17 GT-future-motion val MSE / MAE: `0.031770` / `0.084627`;
  - seed 17 relative reduction: 51.47% MSE, 43.27% MAE.
- Canonical experiment log directory added:
  - `docs/experiments/README.md`;
  - first formal run record:
    `docs/experiments/runs/2026-06-06_gate1_5_four_suite_oracle_action_decoder.md`;
  - reusable template:
    `docs/experiments/templates/run-record-template.md`;
  - policy: `docs/agent_qa/` keeps discussion context, while
    `docs/experiments/` is the cleaned ledger for run configs, metrics,
    artifacts, interpretation, limits, and next decisions.
- SE(3)-aware action metrics added to the oracle action decoder script:
  - code: `scripts/train_oracle_action_decoder.py`;
  - test: `tests/test_oracle_action_decoder_metrics.py`;
  - metrics now include translation, rotation, combined first-six-dim
    `se3`, and gripper MSE/MAE/L2-style decompositions in addition to flat
    MSE/MAE.
- Verification after metric changes:
  - `python -m compileall scripts tests src`;
  - `python -m unittest discover -s tests -p 'test_oracle_action_decoder_metrics.py'`;
  - `python -m unittest discover -s tests -p 'test_libero_hdf5_*.py'`;
  - `PYTHONPATH=/home/user/projects/Geomoco-WM/src python -m unittest discover -s tests`;
  - result: all relevant tests passed; full discover passes 10 tests when
    `src` is on `PYTHONPATH`.
- Gate 1.6 two-file four-suite slice exported:
  - output: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/`;
  - result: 4 suites, 8 HDF5 task files, 400 demos, 8 tasks, 16,518 windows,
    69,073 frames;
  - no dropped short episodes and no exporter warnings;
  - combined `windows.jsonl` size: 68M.
- Gate 1.6 oracle action-decoder comparison completed on CUDA:
  - formal run record:
    `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_se3_metrics.md`;
  - scale-up comparison:
    `docs/experiments/comparisons/2026-06-06_oracle_action_gate_scaleup.md`;
  - split: episode-level;
  - model: MLP `ActionDecoder`, hidden dims `256,256`, 20 epochs,
    batch size 64, `--device cuda`;
  - seed 7 direct-context val MSE / MAE: `0.068109` / `0.148911`;
  - seed 7 GT-future-motion val MSE / MAE: `0.033542` / `0.082386`;
  - seed 7 relative reduction: 50.75% MSE, 44.67% MAE;
  - seed 17 direct-context val MSE / MAE: `0.063910` / `0.145336`;
  - seed 17 GT-future-motion val MSE / MAE: `0.029407` / `0.076629`;
  - seed 17 relative reduction: 53.99% MSE, 47.27% MAE;
  - SE(3) MSE reduction: 82.34% for seed 7, 83.58% for seed 17;
  - gripper MSE reduction: 28.19% for seed 7, 25.26% for seed 17.
- Default restricted execution context did not expose CUDA:
  - `torch 2.10.0+cu128`;
  - `torch.cuda.is_available() == False`;
  - `torch.cuda.device_count() == 0`;
  - this was acceptable for the small CPU MLP gate.
- Elevated GPU checks confirmed the machine and Python environment can see the
  5090:
  - `nvidia-smi`: `NVIDIA GeForce RTX 5090`, driver `580.95.05`, system CUDA
    `13.0`;
  - elevated Python: `torch.cuda.is_available() == True`,
    `cuda_version=12.8`, `device_count=1`;
  - heavier DINO, cVAE, or full-scale training should run from a GPU-visible
    shell or approved execution mode.
- Phase / progress / composition supervision discussion archived:
  - `docs/agent_qa/2026-06-06-phase-progress-composition-supervision.md`
  - decision: keep old GeoMoCo `u_t` as the primary normalized geometric
    motion-progress anchor, define it narrowly as a motion-phase scaffold rather
    than semantic task progress, and add temporal-alignment, gripper/contact,
    visual-change, object-progress diagnostic, and `SE(3)` composition metrics
    as auxiliary probes rather than immediate replacements.
- Dedicated Geomoco-WM uv environment created and installed:
  - venv: `/home/user/projects/Geomoco-WM/.venv`;
  - project cache: `/home/user/projects/Geomoco-WM/.uv-cache`;
  - install command:
    `UV_CACHE_DIR=/home/user/projects/Geomoco-WM/.uv-cache UV_HTTP_TIMEOUT=300 uv pip install -e '.[dev]'`;
  - key versions: `torch 2.12.0+cu130`, `h5py 3.16.0`, `scipy 1.15.3`,
    `pytest 9.0.3`, `ruff 0.15.16`;
  - default restricted shell still reports `torch.cuda.is_available() == False`.
- Action Semantics Audit and geodesic metrics upgrade completed:
  - code:
    `src/geomoco_wm/data/action_semantics.py`,
    `scripts/audit_libero_action_semantics.py`,
    `src/geomoco_wm/metrics/action_metrics.py`;
  - formal record:
    `docs/experiments/runs/2026-06-06_action_semantics_audit_geodesic_metrics.md`;
  - audit artifacts:
    `outputs/action_semantics/libero_four_suite_action_semantics_audit.json`,
    `outputs/action_semantics/libero_four_suite_action_semantics_audit.md`;
  - result: 4 suites, 40 HDF5 files, 2000 demos, no warnings;
  - readiness: all actions are 7D, all controllers are `OSC_POSE`, all use
    delta control, normalized input range is `[-1, 1]`, output scale is
    `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`;
  - metric semantics: translation is now reported after `0.05m` scaling, and
    rotation is now reported as SO(3) geodesic error between scaled rotvec
    exponentials.
- Verification after geodesic metric upgrade:
  - `.venv/bin/python -m compileall src scripts tests`;
  - `.venv/bin/python -m unittest discover -s tests`;
  - `.venv/bin/ruff check ...`;
  - `git diff --check`;
  - result: all checks passed, including 14 unit tests.
- Geodesic metric write-path smoke completed:
  - command: 1-epoch CPU run on
    `outputs/libero_windows/libero_all_suites_smoke/windows.jsonl`;
  - output:
    `outputs/oracle_action_decoder/libero_all_suites_smoke_future_delta_geodesic_1epoch/metrics.json`;
  - final validation examples:
    `val_translation_m_l2=0.043357`,
    `val_rotation_geodesic_rad=0.036910`,
    `val_rotation_geodesic_deg=2.114782`;
  - interpretation: metric plumbing works; this is not a performance result.
- Gate 1.6 geodesic replacement completed on the same 2-files-per-suite slice:
  - formal record:
    `docs/experiments/runs/2026-06-06_gate1_6_two_file_oracle_action_geodesic_replacement.md`;
  - comparison summary:
    `docs/experiments/comparisons/2026-06-06_gate1_6_geodesic_replacement_summary.md`;
  - output metrics:
    `outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed7/metrics.json`,
    `outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/metrics.json`,
    `outputs/oracle_action_decoder/gate1_6_geodesic_direct_seed17/metrics.json`,
    `outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/metrics.json`;
  - mean direct-context validation metrics:
    `val_mse=0.066010`, `val_mae=0.147124`,
    `val_translation_m_l2=0.019024m`,
    `val_rotation_geodesic_deg=2.233651`;
  - mean oracle-future-motion validation metrics:
    `val_mse=0.031474`, `val_mae=0.079508`,
    `val_translation_m_l2=0.007466m`,
    `val_rotation_geodesic_deg=1.048033`;
  - mean reductions from direct context to oracle future motion:
    flat MSE `52.32%`, MAE `45.96%`, translation meter L2 `60.76%`,
    SO(3) geodesic rotation `53.08%`, gripper MSE `26.87%`,
    SE(3) MSE `82.99%`.
- Gate 2 deterministic future-motion prior implemented and run:
  - code:
    `src/geomoco_wm/models/future_motion_predictor.py`,
    `src/geomoco_wm/metrics/motion_metrics.py`,
    `scripts/train_future_motion_predictor.py`;
  - tests:
    `tests/test_future_motion_predictor.py`;
  - formal record:
    `docs/experiments/runs/2026-06-06_gate2_deterministic_future_motion_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-06_gate2_learned_prior_vs_bounds.md`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_deterministic_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_deterministic_seed17/metrics.json`;
  - mean future-motion validation metrics:
    `val_mse=0.001027`, `val_mae=0.019540`,
    `val_l2=0.199044`, `val_translation_l2=0.024785`,
    `val_orientation_coord_l2=0.055226`;
  - mean zero-motion baseline:
    `val_mse=0.001896`, `val_l2=0.250595`,
    `val_translation_l2=0.029872`,
    `val_orientation_coord_l2=0.067285`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.081291`, `action_mae=0.166503`,
    `translation_m_l2=0.022733m`,
    `rotation_geodesic_deg=2.189888`, `gripper_mse=0.296775`;
  - interpretation: learned motion beats zero future motion, but downstream
    action metrics do not beat the direct-context lower bound
    (`action_mse=0.066010`, `translation_m_l2=0.019024m`,
    `rotation_geodesic_deg=2.233651`, `gripper_mse=0.252545`).
  - branch-reading clarification archived in:
    `docs/experiments/runs/2026-06-06_gate2_deterministic_future_motion_prior.md`
    and
    `docs/experiments/comparisons/2026-06-06_gate2_learned_prior_vs_bounds.md`;
    in that clarification, "it" explicitly means the deterministic
    `FutureMotionPredictor(context/proprio) -> predicted future EEF delta`.
- Verification after Gate 2 implementation:
  - `.venv/bin/python -m compileall src scripts tests`;
  - `.venv/bin/python -m unittest discover -s tests`;
  - `.venv/bin/ruff check ...`;
  - `git diff --check`;
  - result: all checks passed, including 16 unit tests.
- Gate 2.1 suite/task-conditioned future-motion prior implemented and run:
  - code:
    `src/geomoco_wm/models/future_motion_predictor.py` now supports optional
    `conditioning_dim`;
    `scripts/train_future_motion_predictor.py` now supports
    `--condition-on none|suite|task|suite_task`;
  - formal record:
    `docs/experiments/runs/2026-06-07_gate2_1_suite_task_conditioned_future_motion_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-07_gate2_1_conditioned_prior_vs_bounds.md`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_1_suite_task_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_1_suite_task_seed17/metrics.json`;
  - conditioning: one-hot `suite_task`, dim 8, built from full dataset
    metadata labels;
  - mean future-motion validation metrics:
    `val_mse=0.000929`, `val_mae=0.018317`,
    `val_l2=0.187877`, `val_translation_l2=0.021964`,
    `val_orientation_coord_l2=0.053070`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.072501`, `action_mae=0.155573`,
    `translation_m_l2=0.020841m`,
    `rotation_geodesic_deg=2.106491`, `gripper_mse=0.279382`;
  - interpretation: suite/task metadata helps compared with Gate 2
    (`action_mse=0.081291` -> `0.072501`), but still does not beat direct
    context (`action_mse=0.066010`), so it remains a diagnostic baseline.
- Gate 2.2a DINOv2 global visual future-motion prior implemented and run:
  - code:
    `src/geomoco_wm/data/visual_feature_cache.py`,
    `scripts/cache_libero_dino_features.py`;
    `src/geomoco_wm/data/window_dataset.py` now supports optional visual
    feature cache attachment;
    `scripts/train_future_motion_predictor.py` now supports
    `--visual-feature-cache`;
  - discussion archive:
    `docs/agent_qa/2026-06-07-visual-grounding-design-and-gate22-plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-07_gate2_2a_dinov2_global_visual_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-07_gate2_2a_visual_prior_vs_bounds.md`;
  - visual cache:
    `outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5`,
    dim 1536 from `dinov2_vits14_reg`, 2 context frames, and two cameras;
  - artifacts:
    `outputs/future_motion_predictor/gate2_2a_dinov2_global_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_2a_dinov2_global_seed17/metrics.json`;
  - mean future-motion validation metrics:
    `val_mse=0.000801`, `val_mae=0.016178`,
    `val_l2=0.171171`, `val_translation_l2=0.016219`,
    `val_orientation_coord_l2=0.050310`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.053628`, `action_mae=0.128207`,
    `translation_m_l2=0.016285m`,
    `rotation_geodesic_deg=2.031589`, `gripper_mse=0.229947`;
  - interpretation: visual grounding produces the first learned prior that
    beats direct context (`action_mse=0.053628` vs `0.066010`) and closes
    `35.85%` of the direct-to-oracle action-MSE gap.
- Gate 2.2b patch-pooled DINOv2 cross-attention visual prior implemented and
  run:
  - code:
    `scripts/cache_libero_dino_features.py` now supports
    `--feature-mode patch_pool`;
    `src/geomoco_wm/models/future_motion_predictor.py` now includes
    `VisualCrossAttentionFutureMotionPredictor`;
    `scripts/train_future_motion_predictor.py` now supports
    `--visual-fusion cross_attention`;
  - formal record:
    `docs/experiments/runs/2026-06-07_gate2_2b_patchpool_cross_attention_visual_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-07_gate2_2_visual_grounding_summary.md`;
  - visual cache:
    `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5`,
    64 visual tokens per window, token dim 384, flat dim 24576;
  - artifacts:
    `outputs/future_motion_predictor/gate2_2b_patchpool4_crossattn_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_2b_patchpool4_crossattn_seed17/metrics.json`;
  - mean future-motion validation metrics:
    `val_mse=0.000772`, `val_mae=0.015710`,
    `val_l2=0.168029`, `val_translation_l2=0.014441`,
    `val_orientation_coord_l2=0.050114`;
  - mean downstream action metrics from predicted motion:
    `action_mse=0.049547`, `action_mae=0.120370`,
    `translation_m_l2=0.014859m`,
    `rotation_geodesic_deg=2.030450`, `gripper_mse=0.222467`;
  - interpretation: patch cross-attention improves action MSE by `7.61%`
    over Gate 2.2a and closes `47.67%` of the direct-to-oracle action-MSE gap.
- Gate 2.2c visual controls implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_2_visual_controls_plan.md`;
  - code:
    `scripts/shuffle_visual_feature_cache.py`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_2c_visual_controls.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_2_visual_controls_summary.md`;
  - visual caches:
    `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5`,
    `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_agentview_2files_h8.h5`,
    `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_eye_in_hand_2files_h8.h5`;
  - shuffled control: `fixed_points=0`, 64 visual tokens per window;
  - single-camera controls: 32 visual tokens per window for `agentview_rgb` or
    `eye_in_hand_rgb`;
  - mean shuffled action metrics:
    `action_mse=0.075521`, `action_mae=0.154999`,
    `translation_m_l2=0.020578m`,
    `rotation_geodesic_deg=2.127742`, `gripper_mse=0.300477`;
  - mean agentview-only action metrics:
    `action_mse=0.054732`, `action_mae=0.127885`,
    `translation_m_l2=0.016336m`,
    `rotation_geodesic_deg=2.014087`, `gripper_mse=0.233136`;
  - mean eye-in-hand-only action metrics:
    `action_mse=0.050853`, `action_mae=0.124119`,
    `translation_m_l2=0.015433m`,
    `rotation_geodesic_deg=2.024622`, `gripper_mse=0.224925`;
  - interpretation: shuffled visual features are worse than direct context,
    while both aligned single-camera branches beat direct context; two-camera
    patch grounding remains best at `action_mse=0.049547`.
- Oracle v2 upper-bound calibration plan archived:
  - plan:
    `docs/experiments/plans/2026-06-08_oracle_v2_upper_bound_and_mainline_order.md`;
  - blueprint update:
    `docs/ideas_plans/plans/geomoco-wm-complete-experiment-blueprint.md`;
  - position: Oracle v2 is a side-track for upper-bound calibration after
    action-aware/multimodal learned priors, not the immediate replacement for
    the learned-prior mainline.
- Gate 2.3a action-aware visual future-motion prior implemented and run:
  - code:
    `scripts/train_future_motion_predictor.py` now supports
    `--action-aware-loss-weight`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_3a_action_aware_visual_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_3a_action_aware_vs_visual_prior.md`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_3a_action_aware_lam001_patchpool4_crossattn_seed17/metrics.json`;
  - training objective:
    `MSE(pred_future_ee_delta, gt_future_ee_delta) + 0.01 * MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)`;
  - mean future-motion validation metrics:
    `val_mse=0.000770`, `val_mae=0.016348`,
    `val_translation_l2=0.016763`,
    `val_orientation_coord_l2=0.049942`;
  - mean downstream action metrics:
    `action_mse=0.043174`, `action_mae=0.113432`,
    `translation_m_l2=0.014835m`,
    `rotation_geodesic_deg=2.037468`, `gripper_mse=0.177930`;
  - interpretation: action-aware loss improves action MSE by `12.86%` over
    Gate 2.2b and raises direct-to-oracle gap closure from `47.67%` to
    `66.12%` without collapsing future-motion MSE.
- Gate 2.3b action-aware lambda sweep completed:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_3b_action_aware_lambda_sweep_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_3b_action_aware_lambda_sweep.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_3_action_aware_lambda_selection.md`;
  - sweep values: `lambda_action in {0.003, 0.010, 0.030}`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_3b_action_aware_lam0003_patchpool4_crossattn_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_3b_action_aware_lam0003_patchpool4_crossattn_seed17/metrics.json`,
    `outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_3b_action_aware_lam003_patchpool4_crossattn_seed17/metrics.json`;
  - mean downstream action metrics for selected lambda `0.030`:
    `action_mse=0.042090`, `action_mae=0.110949`,
    `translation_m_l2=0.014598m`,
    `rotation_geodesic_deg=2.016930`, `gripper_mse=0.174519`;
  - direct-to-oracle action-MSE gap closure:
    `47.67%` at lambda `0.000`, `66.12%` at lambda `0.010`, and
    `69.26%` at lambda `0.030`;
  - tradeoff: lambda `0.030` worsens future-motion translation L2 to
    `0.018767`, compared with `0.016763` at lambda `0.010`;
  - decision: use lambda `0.030` as the action-value-prioritized default and
    keep lambda `0.010` as the balanced geometry reference.
- Gate 2.4a stepwise multi-query visual predictor implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4a_stepwise_multi_query_visual_predictor_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4a_stepwise_multi_query_visual_predictor.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4a_stepwise_vs_single_query.md`;
  - code:
    `StepwiseVisualCrossAttentionFutureMotionPredictor` and
    `--visual-fusion stepwise_cross_attention`;
  - artifacts:
    `outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed7/metrics.json`,
    `outputs/future_motion_predictor/gate2_4a_stepwise_action_aware_lam003_patchpool4_crossattn_seed17/metrics.json`;
  - mean metrics:
    `future_mse=0.000776`, `future_translation_l2=0.017155`,
    `future_orientation_coord_l2=0.051082`, `action_mse=0.042687`,
    `action_mae=0.112300`, `translation_m_l2=0.014750m`,
    `rotation_geodesic_deg=2.042381`, `gripper_mse=0.177896`;
  - interpretation: stepwise queries improve motion-space translation relative
    to single-query lambda `0.030` but do not beat its action MSE
    (`0.042687` vs `0.042090`);
  - decision: keep single-query `cross_attention + lambda_action=0.030` as the
    deterministic action-value default; use stepwise attention only as an
    optional geometry-balanced ingredient.
- Gate 2.4b first visual-conditioned cVAE future-motion prior implemented and
  run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4b_visual_cvae_future_motion_prior_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4b_visual_cvae_future_motion_prior.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4b_visual_cvae_vs_deterministic.md`;
  - code:
    `VisualConditionedGeoMoCoCVAE`,
    `gaussian_kl_divergence`,
    `scripts/train_visual_cvae_future_motion.py`;
  - artifacts:
    `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/metrics.json`,
    `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/metrics.json`;
  - config:
    `latent_dim=32`, `beta_kl=0.001`, `prior_recon_weight=1.0`,
    `action_aware_loss_weight=0.030`;
  - mean prior-mean metrics:
    `prior_mse=0.000802`, `prior_translation_l2=0.017420`,
    `prior_orientation_coord_l2=0.050785`, `kl=0.000740`,
    `action_mse=0.041579`, `action_mae=0.111667`,
    `translation_m_l2=0.014936m`,
    `rotation_geodesic_deg=2.060341`, `gripper_mse=0.165615`;
  - interpretation: weak positive over the deterministic action-value default
    on mean action MSE and gripper MSE, but posterior/prior metrics are almost
    identical and KL is near zero;
  - decision: treat this as a promising cVAE entry point, not proof of
    meaningful multimodal latent usage.
- Gate 2.4c cVAE stochasticity calibration implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4c_cvae_stochasticity_calibration_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4c_cvae_stochasticity_calibration.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4c_cvae_sampling_and_freebits.md`;
  - code:
    `scripts/evaluate_visual_cvae_samples.py`,
    `--beta-kl-start`, `--beta-kl-warmup-epochs`, `--free-bits`,
    and free-bits KL support in `gaussian_kl_divergence`;
  - artifacts:
    `outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/metrics.json`,
    `outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/metrics.json`,
    `outputs/visual_cvae_future_motion/gate2_4c_sample_eval_freebits002_seed7_k16.json`,
    `outputs/visual_cvae_future_motion/gate2_4c_sample_eval_freebits002_seed17_k16.json`;
  - config:
    `latent_dim=32`, `free_bits=0.02`, `beta_kl_start=0.0`,
    `beta_kl=0.001`, `beta_kl_warmup_epochs=5`,
    `prior_recon_weight=1.0`, `action_aware_loss_weight=0.030`;
  - mean metrics:
    `raw_kl=0.442097`, `logged_kl=0.648103`,
    `prior_motion_mse=0.000801`, `sample_motion_mse=0.000847`,
    `best_of_k_motion_mse=0.000552`,
    `sample_motion_variance=0.00004481`,
    `prior_action_mse=0.040931`, `sample_action_mse=0.041199`,
    `best_of_k_action_mse=0.036894`;
  - interpretation: free-bits makes the latent branch non-collapsed and
    improves oracle-selected coverage, but a deployable sample scorer/readout
    is still missing because random sample mean is worse than prior mean;
  - decision: promote free-bits cVAE as the calibrated stochastic branch, not
    as a deployable multimodal policy module yet.
- Gate 2.4d sample scorer/readout discussion and plan archived:
  - discussion:
    `docs/agent_qa/2026-06-08-cvae-sample-scorer-readout-and-related-work.md`;
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4d_cvae_sample_readout_plan.md`;
  - interpretation: published work does have analogous readout mechanisms, but
    the clean first GeoMoCo-WM gate should be a lightweight scorer over frozen
    cVAE samples rather than immediately adopting full Q-guided diffusion or a
    stronger action generator.
- World-model rollout and action-head staging discussion archived:
  - discussion:
    `docs/agent_qa/2026-06-08-world-model-rollout-action-head-and-sample-readout.md`;
  - plan integration:
    `docs/experiments/plans/2026-06-08_gate2_4d_cvae_sample_readout_plan.md`;
  - decision: continue mainline with lightweight ScoreNet first; keep the
    action decoder deterministic until the future-motion sample readout itself
    shows deployable value.
- Gate 2.4d lightweight ScoreNet implemented and run:
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4d_lightweight_cvae_sample_scorer.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4d_sample_readout_vs_oracle.md`;
  - code:
    `src/geomoco_wm/models/sample_readout.py`,
    `scripts/train_visual_cvae_sample_scorer.py`;
  - tests:
    `tests/test_future_motion_predictor.py`;
  - artifacts:
    `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed7/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4d_lightweight_action_rank_k16_seed17/metrics.json`;
  - mean result:
    `prior_action_mse=0.040931`, `sample_mean_action_mse=0.041183`,
    `scorer_argmax_action_mse=0.040201`,
    `oracle_best_action_mse=0.036895`;
  - interpretation: first deployable scorer works and is stable across seeds,
    but the readout is still far from oracle best-of-K.
- Gate 2.4e SE(3)+gripper-aware ScoreNet implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4e_se3_gripper_aware_sample_scorer_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4e_se3_gripper_aware_sample_scorer.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4e_structured_readout_vs_flat.md`;
  - code:
    `scripts/train_visual_cvae_sample_scorer.py`,
    `tests/test_future_motion_predictor.py`;
  - artifacts:
    `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed7/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed17/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed7/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed17/metrics.json`;
  - mean result:
    `flat_action_mse=0.040201`, `se3_action_mse=0.040582`,
    `se3_gripper_action_mse=0.040424`,
    `prior_action_mse=0.040931`, `oracle_best_action_mse=0.036895`;
  - interpretation: naive structured metric replacement is not enough; keep
    the flat action-MSE ScoreNet as the current baseline and move next to
    hard-negative or richer executability/contact supervision.
- Gate 2.4f structured oracle readout evaluation implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4f_structured_oracle_readout_eval_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4f_structured_oracle_readout_eval.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4f_structured_oracle_ranks.md`;
  - code:
    `scripts/train_visual_cvae_sample_scorer.py`,
    `scripts/evaluate_visual_cvae_sample_scorer.py`,
    `tests/test_future_motion_predictor.py`;
  - artifacts:
    `outputs/visual_cvae_sample_scorer_eval/gate2_4f_flat_seed7.json`,
    `outputs/visual_cvae_sample_scorer_eval/gate2_4f_flat_seed17.json`,
    `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_seed7.json`,
    `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_seed17.json`,
    `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_gripper_seed7.json`,
    `outputs/visual_cvae_sample_scorer_eval/gate2_4f_se3_gripper_seed17.json`;
  - mean result:
    `flat_action_mse=0.040190`, `se3_action_mse=0.040642`,
    `se3_gripper_action_mse=0.040441`,
    `flat_rank=6.624765`, `se3_rank=7.289575`,
    `se3_gripper_rank=7.364062`;
  - interpretation: structured scorers move their own structured oracle ranks
    in the expected direction but do not beat flat action-MSE readout, so the
    next branch should add hard negatives or executability supervision rather
    than continue metric-target swaps.
- Gate 2.4g hard-negative readout implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-08_gate2_4g_hard_negative_readout_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-08_gate2_4g_hard_negative_readout.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-08_gate2_4g_hard_negative_vs_flat.md`;
  - code:
    `scripts/train_visual_cvae_sample_scorer.py`,
    `scripts/evaluate_visual_cvae_sample_scorer.py`,
    `tests/test_future_motion_predictor.py`;
  - artifacts:
    `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w01_k16_seed7/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w01_k16_seed17/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w05_k16_seed7/metrics.json`,
    `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w05_k16_seed17/metrics.json`;
  - mean result:
    `flat_score_net_action_mse=0.040190`,
    `hardneg_w01_action_mse=0.040344`,
    `hardneg_w05_action_mse=0.040479`,
    `hardneg_w01_gap_closed=14.55%`,
    `hardneg_w05_gap_closed=11.20%`;
  - interpretation: naive structured-score hard negatives are not reliable
    enough; the next branch should use explicit event/contact/executability
    proxies tied to gripper transitions and action regret.
- Gate 2.4h-a gripper/event label audit implemented and run:
  - plan:
    `docs/experiments/plans/2026-06-09_gate2_4h_phase_event_probe_plan.md`;
  - formal record:
    `docs/experiments/runs/2026-06-09_gate2_4h_a_gripper_event_audit.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-09_gate2_4h_a_command_vs_transition_events.md`;
  - discussion archive:
    `docs/agent_qa/2026-06-09-phase-event-visual-probe-plan.md`;
  - code:
    `src/geomoco_wm/data/event_labels.py`,
    `scripts/audit_gripper_events.py`,
    `tests/test_event_labels.py`;
  - artifacts:
    `outputs/event_audits/gate2_4h_gripper_events_2files.json`,
    `outputs/event_audits/gate2_4h_gripper_events_2files.md`,
    `outputs/event_audits/gate2_4h_gripper_transitions_2files.json`,
    `outputs/event_audits/gate2_4h_gripper_transitions_2files.md`;
  - mean/result summary:
    close sign inferred as `+1` from gripper-width deltas;
    command-state labels have close step concentrated at step 0;
    transition labels produce `878` close transitions, `807` open transitions,
    `5,881` sustain-close windows, and `8,938` sustain-open windows;
  - interpretation: use transition labels, not command-state labels, as the
    event contract for Gate 2.4h-b/c/d.
- Gate 2.4h-b visual phase/event probe implemented and run:
  - formal record:
    `docs/experiments/runs/2026-06-09_gate2_4h_b_visual_phase_event_probe.md`;
  - comparison:
    `docs/experiments/comparisons/2026-06-09_gate2_4h_b_visual_phase_event_probe_summary.md`;
  - code:
    `scripts/train_phase_event_probe.py`,
    `src/geomoco_wm/data/event_labels.py`,
    `tests/test_event_labels.py`;
  - artifacts:
    `outputs/phase_event_probe/gate2_4h_b_<input_variant>_seed7/metrics.json`,
    `outputs/phase_event_probe/gate2_4h_b_<input_variant>_seed17/metrics.json`;
  - mean macro-F1:
    `task_only=0.215345`, `task_proprio=0.421542`,
    `future_motion_only=0.442754`, `proprio_future_motion=0.509978`,
    `visual_only=0.630579`, `visual_proprio_future_motion=0.631385`,
    `shuffled_visual_proprio_future_motion=0.179985`;
  - interpretation: vision is not merely decorative; aligned DINO features
    carry strong manipulation phase/event information, while shuffled visual
    destroys the signal.

## Current Interpretation
- The repository is ready as a clean visual-grounded method track and handoff
  point.
- The current code proves only package structure and minimal tensor plumbing.
  It does not yet prove visual feature export, grounding quality, cVAE training
  stability, predictive value, closed-loop performance, or baseline fairness.
- The project should not claim full world-model status until it predicts or
  samples future motion from visual/proprio/task context and shows value in
  predictive or controlled action-decoder gates.
- Old GeoMoCo evidence should be treated as a warning signal: DINO grounding and
  candidate retrieval may improve representation recall without producing
  closed-loop gains if progress precision, phase estimation, or action execution
  is still the real bottleneck.
- The new blueprint turns that warning into explicit gates: task mining,
  visual-grounding probes, oracle future-motion action decoding, cVAE coverage,
  shared decoder attribution, mechanism closed-loop, and only then
  LIBERO-Long/LIBERO-10 main evaluation.
- The first implementation step is intentionally read-only. It validates HDF5
  field availability and length alignment before committing to an exporter
  schema, which keeps data-side mistakes from leaking into DINO/cache/training.
- Gate 0 passed for `libero_goal`. The first exporter can require
  `agentview_rgb`, `eye_in_hand_rgb`, `ee_states` or `ee_pos + ee_ori`,
  `gripper_states`, `joint_states`, and 7D `actions`. Object-state should not be
  required as a normal model input.
- Gate 1 defines the first executable data contract. For anchor timestep `t`,
  context frames are `[t - context_len + 1, ..., t]`, future EEF target frames
  are `[t + 1, ..., t + horizon]`, and action chunks are
  `[t, ..., t + horizon - 1]`.
- The current exporter materializes numeric targets in JSONL for readability.
  If full-suite JSONL becomes too large, switch targets to HDF5/NPZ while
  preserving the same window index semantics.
- The first experiment script now exists and maps GT future EEF deltas plus
  proprioceptive context to action chunks. This directly serves the oracle
  action-decoder gate before any cVAE or DINO training.
- The first 10-demo oracle smoke is positive: GT future EEF deltas help the MLP
  decoder over direct context on the drawer subset. This supports continuing
  the executable-interface gate, but it is not yet enough to start cVAE claims.
- Cross-paper synthesis now supports a clearer boundary: OASIS is the closest
  EEF-trajectory policy baseline, GuidedVLA is the closest factor-guided
  decoder reference, SDP is the policy-side `SE(3)` geometry baseline, and
  AMPLIFY / ZipMo are motion-prior references. GeoMoCo-WM should borrow their
  interfaces without becoming any one of them.
- Because all four standard LIBERO suites are now local, the oracle-vs-direct
  diagnostic should be promoted from single-task/window-level smoke to
  multi-suite episode-level comparison before cVAE or DINO claims.
- The first interpretable four-suite test should be the small formal slice:
  one HDF5 task file per suite, all demos in those files, horizon 8, stride 4,
  and episode-level train/validation split. Smoke results remain plumbing-only.
- Gate 1.5 is now positive: across two episode-level seeds, GT future EEF
  deltas reduced validation MSE by roughly 51-57% and MAE by roughly 43-47%.
  This supports continuing toward learned future-motion priors, while still not
  proving visual grounding, cVAE sampling, or closed-loop success.
- Gate 1.6 confirms the mechanism at a larger slice: two files per suite still
  gives roughly 51-54% flat MSE reduction and 82-84% SE(3) MSE reduction. The
  oracle future-motion interface is robust enough to promote a learned
  future-motion prior as the next mainline experiment.
- The action metric contract is now confirmed against LIBERO HDF5 metadata and
  local robosuite source behavior. Future action-decoder runs should report
  `translation_m_*`, `rotation_rotvec_rad_*`, and `rotation_geodesic_*` in
  addition to normalized MSE/MAE so results are physically interpretable.
- The Gate 1.6 replacement makes the oracle gap physically readable: the
  mean validation translation action error drops from `0.0190m` to `0.00747m`,
  and the mean SO(3) geodesic rotation error drops from `2.23deg` to `1.05deg`.
  This is the clean reference table for the first learned future-motion prior.
- Gate 2 shows a useful failure mode: context-only deterministic future-motion
  prediction reduces future-delta MSE compared with zero motion, but its
  predicted motion does not improve the downstream action decoder over direct
  context. This suggests the next prior needs stronger conditioning and/or an
  action-aware objective, not merely lower future-motion MSE.
- Gate 2.1 narrows but does not solve that failure mode: task/suite metadata
  reduces future-motion MSE and downstream action MSE, yet the action route is
  still worse than direct context. This suggests the missing signal is not only
  task identity; the prior likely needs visual grounding, temporal structure,
  gripper/contact modeling, or action-aware supervision.
- Gate 2.2a gives the first positive visual-grounding mechanism result:
  DINOv2 global visual features make predicted future EEF motion executable
  enough to beat direct context through the frozen action-decoder interface.
  This supports the GeoMoCo-WM thesis that vision is not merely an auxiliary
  modality, but a grounding signal that turns future motion into a more useful
  policy intermediate variable.
- Gate 2.2b strengthens that result: patch-pooled cross-attention is the best
  learned prior so far, but the gain over global DINO is moderate. This makes
  visual controls mandatory before cVAE: shuffled features should fail, and
  camera ablations should reveal which view carries the value.
- Gate 2.2c confirms visual attribution: shuffled two-camera features drop to
  `action_mse=0.075521`, worse than direct context, while agentview-only
  reaches `0.054732` and eye-in-hand-only reaches `0.050853`. Correct visual
  alignment matters, and the current default branch remains aligned two-camera
  patch cross-attention.
- The remaining gap from Gate 2.2b to oracle future motion is expected because
  oracle future motion is privileged future trajectory information. The gap is
  now a research target, not a negative result: first verify visual attribution
  with controls, then use action-aware or multimodal priors to reduce the
  remaining direct-to-oracle gap.
- Gate 2.3a validates the action-aware hypothesis: optimizing predicted future
  motion through a frozen action decoder improves downstream action value more
  than motion MSE alone. This becomes the new deterministic learned-prior
  baseline before cVAE.
- Gate 2.3b refines that baseline: stronger action-aware supervision at
  `lambda_action=0.030` gives the best downstream action metrics in the current
  sweep, but it should be described as more action-executable rather than more
  geometrically accurate because future-motion translation L2 is worse than the
  `0.010` reference.
- Gate 2.4a shows that query structure alone is not the next bottleneck:
  per-step attention improves future translation geometry but does not improve
  the main action-value metric. The next mainline should therefore move to
  stochastic/multimodal future-motion priors and gripper/contact diagnostics
  instead of another deterministic visual-query ablation.
- Gate 2.4b is a weak positive for the cVAE route: prior mean action MSE is
  slightly better than the deterministic default and gripper MSE improves, but
  the latent is nearly unused. The next cVAE work should focus on stochastic
  coverage and KL calibration, not just another prior-mean score.
- Gate 2.4c confirms the stochastic branch is now using latent capacity:
  free-bits increases raw KL and sample diversity, and best-of-K coverage
  improves both future-motion and downstream action metrics. This is evidence
  that useful futures exist in the sample set, but not yet evidence of a
  deployable policy readout because best-of-K is GT-selected and random sample
  mean is still worse than the prior mean.
- The scorer/readout idea is not ad hoc: it matches several established
  patterns in offline RL and model-based control. For this project, the most
  attribution-clean version is to freeze the calibrated cVAE and train a small
  ranking/energy head before considering heavier diffusion, flow, or Q-guided
  generators.
- The sample readout is the first world-model planning layer in motion space:
  full world models roll out many future states/videos/latents, while the
  current GeoMoCo-WM branch rolls out many future EEF SE(3) motions. Multimodal
  action heads are useful later, but should wait until the motion-rollout
  readout is tested with a deterministic action decoder.
- The PointWorld comparison sharpens the story: PointWorld-style multiplicity
  is primarily planner-side candidate action sequences rolled through an
  action-conditioned 3D point-flow model, whereas GeoMoCo-WM's multiplicity is
  latent future-motion hypotheses generated before action selection. Therefore
  the current project should be described as a visual-grounded, structured,
  multimodal future-motion interface for action, not as a dense 3D world model.
- Compared with nearby latent world-model, stochastic video, diffusion planner,
  CVAE-action, and candidate-action methods, GeoMoCo-WM's potential advantage
  is the intermediate rollout space: it is narrower than pixels/dense 3D and
  more interpretable than generic latent state, while avoiding the attribution
  ambiguity of direct strong action heads. Gate 2.5d showed this space is real
  but the current flat scorer is still too weak to close the oracle gap.
- Gate 2.4d validates the readout direction but exposes the next bottleneck:
  action-distance ranking alone can select better samples than prior mean, but
  not enough of the best-of-K coverage. The next scorer should include
  gripper/contact/executability signals or hard-negative ranking before moving
  to diffusion/flow action heads.
- Gate 2.4e shows that simply decomposing the scorer target into
  meter-scaled translation, SO(3) geodesic rotation, and gripper MSE does not
  solve the readout bottleneck. The useful next step is not another metric-unit
  target swap, but richer ranking supervision: hard negatives, contact proxies,
  execution feasibility, or direct regret/rank prediction.
- Gate 2.4f confirms that conclusion under cleaner oracle diagnostics:
  structured scorers do improve the structured rank they target, but still do
  not beat the flat scorer on deployable action MSE. Treat `SE(3)` and
  `SE(3)+gripper` ranks as diagnostics, not promotion metrics.
- Gate 2.4g confirms the first hard-negative variant is too naive: choosing
  negatives by structured `SE(3)+gripper` score alone can fight the useful flat
  action ranking signal. The next readout improvement should introduce
  explicit event or executability labels rather than another scalar-score
  contrast.
- Gate 2.4h-a confirms event timing is measurable but must be defined as
  transitions, not command states. This gives the project a cleaner GeoMoCo
  phase/composition probe: close/open phase boundaries can now be tested
  against future-motion samples and visual grounding without immediately
  adding another scorer loss.
- Gate 2.4h-b confirms visual grounding helps the phase/composition probe:
  visual-only and visual+proprio+future-motion probes outperform task/proprio
  and future-motion-only probes, while shuffled visual collapses. This supports
  the claim that visual context identifies manipulation phase and future motion
  expresses candidate phase transitions.
- Oracle v2 remains useful, but only as upper-bound calibration. Use it when
  learned priors approach the current oracle bound or progress stalls and we
  need to know whether missing gripper/contact/object/decoder information is
  limiting the oracle itself.
- The metric decomposition suggests a split modeling plan: future EEF motion is
  the primary geometric branch, while gripper/contact should be modeled or
  supervised separately because its gains are positive but much smaller.
- Future formal experiment results should be recorded under
  `docs/experiments/runs/`, with cross-run ablations and promotion decisions
  under `docs/experiments/comparisons/`.
- Phase, progress, and composition should be reported separately. `u_t` remains
  the lowest-cost stable coordinate for segment construction and motion-phase
  probing, while semantic progress should be treated as task-specific and
  diagnostic unless clean object/contact/success labels exist.
- GeoMoCo-WM should evaluate new progress heuristics by mechanism value, not by
  reconstruction alone: future state/progress prediction, composition
  hard-negative ranking, future-motion coverage, and the motion-to-action
  decoder gap are the promotion gates.

## Open Decisions Or Blockers
- Run heavy DINO, cVAE, or large full-suite training from a GPU-visible shell or
  approved execution mode; the default restricted context hides CUDA even
  though elevated Python can see the 5090.
- Decide whether JSONL target materialization is acceptable for the next reader,
  or whether to move numeric arrays to HDF5/NPZ before full export.
- Full four-suite export remains optional; the next mainline experiment can
  start with the 2-files-per-suite slice because the oracle interface already
  scaled positively.
- Treat object-state teacher fields as unavailable for `libero_goal`; only add
  object-state upper bounds if a later suite/source provides them.
- Define the exact visual-grounded dataset contract, including RGB history,
  DINO feature cache, proprioception, EEF pose, gripper, task, future motion,
  action chunk, and optional object-state teacher fields.
- Task/suite conditioning alone was tested and is not sufficient; DINO global
  and patch visual grounding are positive, and Gate 2.2c visual controls now
  pass. Gate 2.3 action-aware training is positive and lambda `0.030` is the
  current action-value default; the next blocker is whether multi-query
  attention, stochastic/multimodal priors, and gripper/contact modeling can
  further reduce the remaining oracle gap before cVAE claims.
- Next-step sequencing decision archived in
  `docs/agent_qa/2026-06-08-oracle-gap-and-next-step-sequencing.md`: visual
  controls come first, then action-aware/multimodal gap reduction if controls
  pass.
- Define the first `u_geom` target builder for exported LIBERO windows and
  decide whether phase bins / temporal-rank pairs are generated during export
  or in a separate target-materialization pass.
- Choose DINO backbone/version and feature-cache format.
- Decide the first action decoder: simple action-chunk transformer first, with
  Diffusion Policy / MeanFlow-style decoder only after attribution is clear.
- The small formal oracle future-motion diagnostic passed; if a larger-scale
  run fails to beat direct BC, redesign the interface or task suite before
  spending more effort on the cVAE-to-action route.
- Decide the concrete LIBERO-Long/LIBERO-10 task subset after a headroom audit:
  avoid tasks where direct DINO/BC is saturated, and avoid tasks where all
  policies fail for unrelated execution reasons.
- Decide whether object state is teacher/diagnostic only or included in any
  upper-bound baseline.
- Project dev dependencies are installed in the dedicated `.venv`; use
  `.venv/bin/python -m unittest discover -s tests` and `.venv/bin/ruff check`
  for local checks.

## Next Session Entry Point
1. Start Gate 3.2i decision slice: either build a stronger temporal
   boundary-localization objective/head that can approach the Gate 3.2h oracle
   mask, or formally pivot to a richer temporal/flow action decoder for
   gripper transitions.
2. Keep Gate 3.1f/g full event/rank/prob top-4 as the current deployable
   reference: action MSE `0.034767`, gripper MSE `0.150052`; Gate 3.2a
   reproduces this with action MSE `0.034773`. Gate 3.2f/g/h do not beat it as
   deployable predicted-mask heads.
3. Preserve repeated stochastic eval, group audit, and boundary-quality audit.
   Gate 3.2h showed oracle boundary masks are strong (`0.032018` overall MSE)
   but predicted masks are not (`~0.035201` best predicted-mask MSE).
4. Do not add another sparse CE residual variant. Any deterministic continuation
   should change the temporal localization objective/head, not just threshold
   or reweight the same boundary classifier.
5. Do not mix Gate 3 action-head metrics with old frozen-action-decoder metrics;
   use them as separate evidence streams.
6. For heavy training, use a GPU-visible shell or approved execution mode; the
   default restricted context hides CUDA while elevated Python sees the RTX
   5090 correctly.
