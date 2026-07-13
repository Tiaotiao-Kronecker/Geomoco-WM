# cVAE Loss, KL, Free Bits, Beta Schedule, And Eval Contract

- Date: 2026-06-10
- Context: Gate 2.5d/2.5e joint GeoMoCo-cVAE readout discussion

## 1. Are We Choosing Loss Or Eval?

The current mainline is not changing the evaluation target every time a new
readout objective is tested.

The fixed evaluation contract is:

```text
candidate future motion
  -> frozen action decoder
  -> predicted action chunk
  -> action MSE / action metrics against GT action
```

Event metrics are auxiliary diagnostics:

```text
event accuracy
transition accuracy
transition step within 1
event alignment error
```

So the current experiments are:

```text
choose / test different training losses or readout targets
evaluate all of them under the same action-value contract
```

In Gate 2.5e, event-aware ScoreNet variants improved event metrics but worsened
the primary action-MSE evaluation. Therefore they are not promoted.

This avoids the trap:

```text
train a model on event loss
then declare success only because event eval improved
while action-value eval regressed
```

## 2. cVAE Loss Versus ScoreNet Loss

There are two separate training stages.

### GeoMoCo-cVAE loss

The cVAE learns to generate candidate futures:

```text
current visual/proprio/task context
  -> future_delta_ee + future_gripper/event distribution
```

Its loss decides whether good future candidates exist in the sample set.

### ScoreNet loss

ScoreNet is trained after the cVAE is frozen:

```text
cVAE samples K future motions
ScoreNet selects one sample or ranks them
```

Its loss decides whether we can choose a useful sample from the existing
candidate set.

Therefore:

```text
cVAE loss:
  can the model generate useful candidate futures?

ScoreNet loss:
  can the readout choose the useful candidate?
```

## 3. Current GeoMoCo-cVAE Loss

Current cVAE training loss:

```text
L_cVAE =
  L_posterior_recon
  + λ_prior * L_prior_recon
  + β_KL * KL(q(z | c, m_gt) || p(z | c))
  + λ_action * L_action
```

Where:

```text
c = visual + proprio/context + suite/task condition
m_gt = future motion target
```

In the current mainline:

```text
m_gt = future_delta_gripper
     = future_delta_ee + future_gripper/event
```

### Posterior reconstruction

```text
L_posterior_recon = MSE(decoder(c, z_q), m_gt)
```

The posterior sees the real future motion during training. This branch teaches
the latent space how to represent the actual future.

### Prior mean reconstruction

```text
L_prior_recon = MSE(decoder(c, μ_prior), m_gt)
```

The prior sees only current context. This branch makes the deployable prior mean
predict a useful future without seeing the answer.

### KL loss

```text
KL(q(z | c, m_gt) || p(z | c))
```

This aligns the training-time posterior with the test-time prior.

### Action-aware loss

```text
L_action =
  MSE(ActionDecoder(context, prior_mean_reconstruction), action_gt)
```

The action decoder is frozen. This makes prior-mean future motion more useful
as a downstream action interface.

## 4. What Are q And p?

The posterior distribution is:

```text
q(z | c, m_gt)
```

It is used during training and sees:

```text
current context c
ground-truth future motion m_gt
```

It answers:

```text
given the actual future, what latent z explains it?
```

The prior distribution is:

```text
p(z | c)
```

It is used at test time and sees only:

```text
current context c
```

It answers:

```text
given only the current observation, what future latents are plausible?
```

Training has both q and p. Deployment has only p.

The KL term:

```text
KL(q || p)
```

pushes the deployable prior to match the posterior's answer-informed latent
distribution.

## 5. KL Formula

Both q and p are diagonal Gaussians:

```text
q = N(μ_q, σ_q^2)
p = N(μ_p, σ_p^2)
```

The KL is:

```text
KL(q || p)
= 1/2 * Σ_i [
    log(σ_p_i^2 / σ_q_i^2)
    + (σ_q_i^2 + (μ_q_i - μ_p_i)^2) / σ_p_i^2
    - 1
  ]
```

Low KL can mean the posterior and prior are well aligned, but it can also mean
the latent is being ignored. High KL means the posterior is encoding information
the prior may not be able to reproduce at test time.

## 6. What Is Beta Schedule?

The cVAE loss uses:

```text
L = recon_loss + β * KL
```

`β` controls how strongly the model is forced to align posterior and prior.

Beta schedule means:

```text
early training:
  β is small, so the model first learns reconstruction

later training:
  β increases toward the target value, so prior and posterior align
```

Intuition:

```text
first learn to express the future
then learn to generate it from the prior
```

## 7. What Are Free Bits?

Free bits give each latent dimension a small KL allowance:

```text
KL_total = Σ_i max(KL_i, τ)
```

where `τ` is the free-bits threshold.

This reduces the pressure to drive each latent dimension's KL all the way to
zero. It helps prevent posterior collapse, where:

```text
q ≈ p
z carries little information
the cVAE behaves like a deterministic predictor
```

In the current joint cVAE mainline, the useful calibration has been around:

```text
free_bits = 0.02
beta_kl = 0.001
beta warmup = 5 epochs
```

## 8. Current Working Interpretation

The current cVAE best-of-K results show that the sample set contains useful
future candidates. The harder problem is readout:

```text
can we choose a deployable sample without GT best-of-K selection?
```

Gate 2.5e showed that event-aware readout can improve event metrics, but that
does not automatically improve action-value. This means event timing is a real
signal, but the next mainline should keep the action-eval contract fixed and
improve either:

```text
the joint cVAE sample space
or
the temporal/action-regret readout
```

rather than simply increasing event loss weight.

