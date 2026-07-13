# Gate 2.4c cVAE Sampling And Free-Bits Comparison

- Date: 2026-06-08
- Status: completed
- Scope: compare Gate 2.4b visual cVAE prior samples against the Gate 2.4c
  free-bits calibrated cVAE.

## Summary Table

| branch | raw KL | logged KL | prior motion MSE | sample motion MSE | best-of-K motion MSE | sample var | sample pair L2 | prior action MSE | sample action MSE | best-of-K action MSE | prior grip MSE | best-of-K grip MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.4b visual cVAE | 0.000740 | 0.000740 | 0.000802 | 0.000803 | 0.000744 | 0.00000167 | 0.012125 | 0.041579 | 0.041622 | 0.039526 | 0.165615 | 0.158352 |
| Gate 2.4c free-bits cVAE | 0.442097 | 0.648103 | 0.000801 | 0.000847 | 0.000552 | 0.00004481 | 0.055588 | 0.040931 | 0.041199 | 0.036894 | 0.167670 | 0.152207 |

## Readout

Free-bits turns the cVAE from a near-deterministic prior into a genuinely
used latent branch:

```text
raw KL: 0.000740 -> 0.442097
sample variance: 26.90x
sample pair L2: 4.58x
```

Coverage improves:

```text
best-of-K motion MSE improves by 25.83%
best-of-K action MSE improves by 6.66%
```

The deployable prior mean also improves, but only slightly:

```text
prior-mean action MSE: 0.041579 -> 0.040931
relative improvement: 1.56%
```

The random sample mean remains worse than the prior mean:

```text
Gate 2.4c prior-mean action MSE: 0.040931
Gate 2.4c sample-mean action MSE: 0.041199
```

So the stochastic branch now contains useful samples, but choosing them still
requires a readout or ranking mechanism.

## Decision

Promote Gate 2.4c as the calibrated cVAE branch for future stochastic work.

Do not yet claim deployable multimodal policy value. The evidence is:

- positive for latent usage and oracle-selected coverage;
- mildly positive for prior mean;
- not yet positive for naive random sampling;
- incomplete for gripper/contact.

## Mainline Implication

The next branch should not simply increase `K` and report best-of-K again.
Instead, it should test how to turn the sampled future-motion set into a
deployable policy interface:

1. learned sample scorer or energy head conditioned on context and visual
   tokens;
2. action-aware risk aggregation through the frozen decoder;
3. gripper/contact auxiliary targets;
4. closed-loop or policy-facing validation once the readout is credible.
