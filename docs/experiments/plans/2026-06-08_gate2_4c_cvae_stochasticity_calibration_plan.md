# Gate 2.4c cVAE Stochasticity Calibration Plan

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4c
- Purpose: test whether the visual-conditioned cVAE learned meaningful
  stochastic / multimodal future-motion coverage, and whether KL calibration
  can improve latent usage.

## Motivation

Gate 2.4b was a weak positive on prior-mean action value:

```text
deterministic single-query lambda 0.030 action MSE: 0.042090
visual cVAE prior-mean action MSE: 0.041579
```

But KL was near zero:

```text
mean KL: 0.000740
```

Posterior and prior reconstructions were nearly identical, so we cannot yet
claim that the cVAE learned a meaningful multimodal future-motion distribution.

## Questions

Gate 2.4c asks four concrete questions:

1. Do prior samples differ meaningfully from the prior mean?
2. Does best-of-K sampling improve future-motion coverage over the prior mean?
3. Does best-of-K action selection improve downstream action metrics?
4. Can KL free-bits or beta scheduling increase latent usage without hurting
   deployable prior quality?

## Dataset

Use the same four-suite formal slice as Gates 2.3b / 2.4a / 2.4b:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

This covers:

```text
4 suites, 8 selected task files, 400 demos, 16,518 windows
```

## Part A: Sample Evaluation Of Existing Gate 2.4b

Evaluate both existing seed checkpoints:

```text
outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/model.pt
outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/model.pt
```

Metrics:

- prior mean future-motion metrics;
- mean random prior-sample future-motion metrics;
- best-of-K future-motion metrics, selected by GT future-motion MSE;
- best-of-K action metrics, selected by GT action MSE through frozen decoder;
- sample diversity:
  - sample motion variance;
  - sample-to-sample L2;
  - distance from prior mean.

Important: best-of-K uses GT for selection, so it is a coverage diagnostic, not
a deployable policy result.

## Part B: KL Calibration

Add training switches:

```text
--beta-kl-start
--beta-kl-warmup-epochs
--free-bits
```

Initial calibration matrix:

| branch | beta schedule | free bits | reason |
| --- | --- | ---: | --- |
| Gate 2.4b baseline | fixed `0.001` | `0.0` | existing reference |
| Gate 2.4c free-bits | fixed `0.001` | `0.02` | reduce pressure for posterior-prior collapse |
| Optional beta warmup | `0 -> 0.001` | `0.0` | test whether posterior can learn before KL pressure |

Run primary free-bits branch with seeds `7` and `17` if the implementation
passes smoke.

## Pass Criteria

The stochastic route becomes credible only if:

- best-of-K future-motion metrics clearly beat prior mean;
- sample diversity is non-trivial;
- KL is meaningfully above the near-zero baseline;
- prior mean action metrics do not collapse relative to Gate 2.4b;
- ideally, best-of-K action coverage indicates that some samples are closer to
  the oracle action route.

## Stop Criteria

Do not claim stochastic / multimodal value if:

- samples have near-zero diversity;
- best-of-K is almost identical to prior mean;
- KL remains near zero;
- free-bits increases KL but hurts prior mean badly;
- action gains appear only in GT-selected best-of-K without deployable
  improvement.

## Expected Mainline Decision

If Gate 2.4c shows no useful sample coverage, the project should treat the
current cVAE as a deterministic regularized prior and move to gripper/contact
diagnostics or stronger action decoders before claiming multimodal GeoMoCo-WM.

If Gate 2.4c is positive, the next gate should integrate sample selection /
risk-sensitive action readout and compare against deterministic baselines.

## Execution Result

Gate 2.4c completed with two parts:

1. `K=16` sample evaluation of the existing Gate 2.4b checkpoints.
2. free-bits cVAE retraining with `free_bits=0.02`,
   `beta_kl_start=0.0`, `beta_kl=0.001`, and
   `beta_kl_warmup_epochs=5`.

Mean result:

```text
Gate 2.4b raw KL: 0.000740
Gate 2.4c raw KL: 0.442097

Gate 2.4b best-of-K action MSE: 0.039526
Gate 2.4c best-of-K action MSE: 0.036894

Gate 2.4b prior-mean action MSE: 0.041579
Gate 2.4c prior-mean action MSE: 0.040931
```

Decision: Gate 2.4c passes as a stochasticity calibration gate, but not as a
deployable stochastic policy claim. The next step is a sample readout / scoring
gate, plus gripper/contact diagnostics.

Formal records:

- `docs/experiments/runs/2026-06-08_gate2_4c_cvae_stochasticity_calibration.md`
- `docs/experiments/comparisons/2026-06-08_gate2_4c_cvae_sampling_and_freebits.md`
