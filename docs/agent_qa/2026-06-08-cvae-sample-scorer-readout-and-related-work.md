# cVAE Sample Scorer / Readout Explanation And Related Work

- Date: 2026-06-08
- Context: after Gate 2.4c cVAE stochasticity calibration.

## User Question

为什么说：

```text
随机 sample 平均仍然不如 prior mean，所以下一步瓶颈是 sample scorer/readout，不是继续单纯报 best-of-K。
```

以及已发表工作里是否有类似的 scorer / readout？

## Local Explanation

Gate 2.4c 的结果说明：

```text
cVAE prior samples 里已经有更好的 future-motion 候选；
但实际推理时，我们还不知道该选哪一个。
```

当前有三种读数：

1. `prior mean`
   - 不采样，直接使用 `p(z | context, vision)` 的均值；
   - 稳定、可部署；
   - Gate 2.4c action MSE: `0.040931`。
2. `random sample mean`
   - 从 prior 里随机采样多个 `z_k`，逐个 decode；
   - 统计随机样本平均表现；
   - Gate 2.4c action MSE: `0.041199`，比 prior mean 差。
3. `best-of-K`
   - 从 `K=16` 个样本里，用 GT future motion 或 GT action 选最好的；
   - Gate 2.4c action MSE: `0.036894`；
   - 这是 coverage diagnostic，不是 deployable policy metric。

因此当前结论是：

```text
sample set 里有好东西；
但没有不用 GT 的机制把好东西挑出来。
```

这就是 sample scorer / readout 的位置：

```text
context + vision
  -> cVAE prior
  -> K future-motion samples
  -> scorer/readout(context, vision, sample, optional decoded action)
  -> selected or weighted future motion
  -> action decoder / policy head
```

## Related Work Taxonomy

### 1. Prior Mean As Readout: ACT

ACT / ALOHA uses a CVAE-style action-chunk model to handle non-stationary and
multimodal demonstrations, but common inference uses the prior mean / zero
latent rather than a learned sample selector.

Reference:

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)

Implication for GeoMoCo-WM:

- `prior mean` is a legitimate baseline, not a mistake.
- But if our evidence comes from best-of-K samples, ACT-style prior mean is not
  enough to claim stochastic deployment value.

### 2. Candidate Generator Plus Value / Q Scorer: BCQ

BCQ is a very close algorithmic ancestor of the idea:

```text
state -> VAE generates candidate actions
candidate actions -> Q critic
select argmax_Q
```

Reference:

- [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900)

Implication for GeoMoCo-WM:

- Replace BCQ's action VAE with our future-motion cVAE.
- Replace action-level Q with a future-motion/action executability scorer:

```text
score(c_t, future_motion_sample, decoded_action)
```

This is the most direct precedent for `K samples + scorer`.

### 3. Energy-Based Action Selection: IBC

Implicit Behavioral Cloning learns an energy model over observation-action
pairs and performs inference by optimizing or sampling for low-energy actions.

Reference:

- [Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137)

Implication for GeoMoCo-WM:

- A scorer can be trained as an energy:

```text
E(context, vision, future_motion, decoded_action)
```

- Lower energy means "more demonstration-like / executable".
- This is attractive because LIBERO demonstrations provide positive examples,
  and cVAE samples can provide hard negatives.

### 4. Beam Search / Trajectory Scoring: Trajectory Transformer

Trajectory Transformer models full state-action-reward sequences and uses beam
search as a planning/readout mechanism. It can score candidate continuations by
model likelihood, reward prediction, or a Q-function in harder sparse-reward
settings.

Reference:

- [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)

Implication for GeoMoCo-WM:

- Our future-motion samples can be treated as candidate short trajectories.
- A beam/search-style readout is plausible once we predict multi-step future
  motion and maybe gripper/contact.

### 5. Diffusion Planning With Reward / Energy Guidance

Diffuser generates trajectories through denoising and can guide samples with
reward gradients or constraints.

Reference:

- [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)

QGPO / CEP extends this idea by learning exact energy-guided diffusion sampling
for offline RL.

Reference:

- [Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning](https://arxiv.org/abs/2304.12824)

Implication for GeoMoCo-WM:

- Instead of post-hoc selecting among cVAE samples, one can guide the generation
  distribution itself toward high-score futures.
- This is stronger but more complex than a first readout gate.

### 6. Return-Conditioned Generation: Decision Diffuser

Decision Diffuser conditions trajectory generation on return / constraints /
skills, making the readout more implicit: ask the generator for a high-return
trajectory rather than generate many and rank afterward.

Reference:

- [Is Conditional Generative Modeling all you need for Decision-Making?](https://arxiv.org/abs/2211.15657)

Implication for GeoMoCo-WM:

- If LIBERO success/progress labels become reliable, condition future-motion
  generation on progress/success level.
- This is probably a later branch, because current LIBERO HDF5 windows do not
  yet provide a clean dense reward.

### 7. Visual MPC / CEM Readout

Visual Foresight and PETS use model predictive control: generate candidate
action sequences, predict futures, evaluate them with a task cost/reward, then
execute the first action.

References:

- [Visual Foresight](https://arxiv.org/abs/1812.00568)
- [PETS](https://arxiv.org/abs/1805.12114)

Implication for GeoMoCo-WM:

- This is a hand-designed scorer route:

```text
score = visual goal cost + motion smoothness + contact feasibility
```

- It is useful when task goals are explicit, but less clean for all LIBERO
  tasks unless we define reliable goal/progress costs.

### 8. Motion Prior Plus Inverse Dynamics: AMPLIFY

AMPLIFY separates visual motion prediction from action inference: a forward
dynamics model learns motion tokens from action-free videos, and an inverse
dynamics model maps motion to robot actions.

Reference:

- [AMPLIFY: Actionless Motion Priors for Robot Learning from Videos](https://arxiv.org/abs/2506.14198)

Implication for GeoMoCo-WM:

- AMPLIFY is not exactly a sample scorer, but it supports our decomposition:

```text
predict useful motion first;
then ask whether that motion is executable as robot action.
```

- A future GeoMoCo-WM scorer can borrow this idea by scoring samples through
  inverse-dynamics executability.

### 9. Diffusion Policy

Diffusion Policy directly models the action distribution with conditional
denoising and uses receding-horizon control. It does not need a separate
post-hoc scorer in the same way, because denoising itself is the learned
readout from noise to an action trajectory.

Reference:

- [Diffusion Policy](https://arxiv.org/abs/2303.04137)

Implication for GeoMoCo-WM:

- Diffusion/flow action heads are strong later baselines.
- But they blur the separation we currently want to test:

```text
future-motion prior value
vs.
strong action generator value
```

For the next gate, keep a simple scorer first.

## Main Conclusion

Yes, published work has several versions of this idea:

- BCQ: VAE candidates + Q scorer.
- IBC: energy scorer over actions.
- Trajectory Transformer: beam/search readout over trajectories.
- Diffuser / QGPO: reward or energy-guided generation.
- Visual MPC / PETS: candidate trajectories + cost/reward scoring.

For GeoMoCo-WM, the cleanest next step is not to immediately adopt full
Diffusion Policy or QGPO. The clean first gate is:

```text
Gate 2.4d:
  freeze calibrated cVAE
  sample K future motions
  train a lightweight scorer/readout
  compare:
    prior mean
    random sample
    oracle best-of-K
    learned scorer selected sample
```

This directly tests whether the stochastic future-motion set can be converted
into a deployable policy interface.
