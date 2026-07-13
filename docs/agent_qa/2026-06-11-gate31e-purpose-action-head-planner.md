# Gate 3.1e Purpose: Predicted Event Mixture Into Action Head

## Question

Why should Gate 3.1e feed the predicted top-4 event-mixture sample set into the
Gate 3 action head/planner protocol?

## Answer

Gate 3.1d showed that predicted top-4 event modes place good
`future_delta_gripper` candidates inside the sample set:

```text
predicted top-4 best-of-K action MSE: 0.015228
oracle-event best-of-K action MSE: 0.014656
```

But best-of-K is an oracle diagnostic because it uses ground-truth action error
to choose the best candidate after sampling. It proves the sample space has
good futures; it does not prove the deployed system can choose or use them.

Gate 3.1e therefore asks the deployable question:

```text
current visual/proprio context
-> event predictor
-> top-4 event modes
-> event-conditioned cVAE samples
-> action head/planner
-> action chunk
```

## Comparison Contract

| Comparison | Purpose |
| --- | --- |
| unconditional sample set | tests whether explicit event structure adds value beyond ordinary cVAE sampling |
| shuffled/control | checks that gains come from aligned visual event prediction, not generic diversity |
| oracle-event upper bound | measures remaining headroom from event prediction and sample consumption |
| context-only / prior mean | checks whether the sample set helps beyond no-multimodal-future inputs |

## Mainline Meaning

If Gate 3.1e succeeds, the story becomes:

```text
GeoMoCo-WM proposes a visually grounded, multimodal, event-structured
future-motion set; a downstream action head/planner can consume that set for
better action prediction.
```

If it fails, Gate 3.1d is still meaningful: good futures exist in the candidate
set, but the current action head/planner cannot reliably consume a wide and
noisy predicted-event mixture.

## Guardrail

Do not feed oracle event labels into the deployable Gate 3.1e action head.
Oracle-event conditioning remains an upper-bound diagnostic only.
