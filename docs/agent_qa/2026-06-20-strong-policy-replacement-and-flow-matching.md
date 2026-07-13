# Strong-Policy Replacement Risk And Flow-Matching Next Step

Date: 2026-06-20

## Question

If a strong policy can absorb the gains from GeoMoCo-WM, does that mean the
GeoMoCo-WM benefit is replaceable by diffusion-policy-style action models?

## Short Answer

Yes, this is a real possibility and should be treated as a main hypothesis to
test, not as a side concern.

If a strong policy sees the same observation/history and learns the same or
better action-chunk distribution without explicit GeoMoCo-WM samples, then the
in-distribution performance value of the GeoMoCo-WM interface is not proven in
that setting. The motion prior may still have value as an inductive bias,
diagnostic tool, sample-efficiency mechanism, OOD/compositional aid, or
interpretable planning interface, but it would not be necessary for that
particular strong-policy metric.

## Why This Matters Now

Recent gates make the issue sharper:

```text
Gate 3.4: GeoMoCo samples + event metadata improve the controlled temporal decoder.
Gate 3.5b: post-hoc residual capacity improves overall MSE, but context-only is close.
Gate 3.7a: soft event-time latent worsens temporal/transition metrics.
Gate 3.8a: tiny temporal-transformer action-chunk readout worsens transition/gripper.
```

The pattern says that small decoder changes do not reliably solve the
transition/open-close bottleneck. It also warns that stronger decoders may make
the prior contribution harder to attribute.

Therefore the main question should become:

```text
Under a stronger action policy, does aligned GeoMoCo-WM conditioning still
provide a measurable edge over a direct visual policy?
```

## Why Flow Matching Instead Of Standard Diffusion First

Standard Diffusion Policy is a strong and relevant baseline for action chunks,
but it introduces heavier sampling machinery and many denoising steps.

For this project's low-dimensional action chunks, flow matching is a lighter
first strong-policy testbed:

```text
x1 = ground-truth action chunk [H,A]
x0 = Gaussian noise
t  ~ Uniform(0, 1)
xt = (1 - t) x0 + t x1
target velocity = x1 - x0
model predicts v_theta(xt, t, condition)
```

Inference can use a small Euler integration budget, for example 8 or 16 steps.
This should be enough to test whether a generative action-chunk policy can
replace the current GeoMoCo-WM action-head gains.

## Attribution Rule

Flow matching helps efficiency and implementation simplicity. It does not solve
attribution by itself.

The required comparisons remain:

```text
replacement gap = direct_visual_flow - full_geomoco_flow
prior gain      = no-prior/context-only - full_geomoco_flow
metadata gain   = shuffled/rank-prob-only - full_geomoco_flow
diversity gain  = mean_repeated - full_geomoco_flow
```

Use MSE reductions as positive gains.

## Decision

Proceed to a narrow Gate 3.9a:

```text
controlled conditional flow-matching action policy audit
```

First run only:

```text
direct_visual_flow seed7
geomoco_conditioned_flow seed7
```

If GeoMoCo-conditioned flow does not beat direct visual flow, stop and do not
expand controls. If it does beat direct visual flow, expand to seed17 and then
the attribution controls.
