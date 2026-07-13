# Gate 3.5b Post-Hoc Adapter Motivation

## Question

为什么 Gate 3.5a 失败后，不直接上更大的 flow/diffusion decoder，而要先做
Gate 3.5b：冻结已 promoted 的 Gate 3.4 checkpoint，只训练一个 post-hoc
residual adapter？

## Short Answer

Gate 3.5a 同时改变了两件事：

```text
1. 原来的 Gate 3.4 temporal decoder 继续被新 objective 更新；
2. 新增 flow residual branch 也被一起训练。
```

所以它的失败有归因歧义。结果显示：

```text
Gate 3.4 temporal MSE: 0.034262
Gate 3.5a temporal MSE: 0.035392
Gate 3.5a flow MSE:     0.035621
```

这说明不只是 flow residual 没赢，连同 checkpoint 里的 temporal branch 本身也
比 Gate 3.4 退化了。因此 Gate 3.5a 更像是在说明“joint residual-flow
training 会干扰共享表示/temporal decoder”，而不是证明“residual action
modeling 没有价值”。

## What Post-Hoc Means

`post-hoc adapter` 指的是：

```text
先把 Gate 3.4 action head 完全冻结；
每个 batch 用冻结模型产生 features 和 temporal_actions；
再训练一个很小的新 adapter 预测 residual；
最终输出 adapter_actions = frozen_temporal_actions + residual。
```

它不是替换 Gate 3.4，也不重新训练 GeoMoCo-WM prior、event probe、sample
interface 或 temporal decoder。它只回答一个更窄的问题：

```text
在 Gate 3.4 已经学好的表示和 temporal_actions 之上，
一个小 residual sequence adapter 是否还能稳定改进动作轨迹？
```

## Difference From Gate 3.5a

Gate 3.5a:

```text
frozen? no
训练对象: action head shared features + temporal decoder + flow residual
风险: 新 residual objective 会把原本 promoted 的 Gate 3.4 temporal branch 带坏
decoder gain: same-checkpoint temporal_actions - same-checkpoint flow_actions
```

Gate 3.5b:

```text
frozen? yes, Gate 3.4 action head 完全冻结
训练对象: only post-hoc residual adapter
风险: adapter 自己可能没用，但不会破坏 Gate 3.4
decoder gain: frozen Gate 3.4 temporal_actions - adapter_actions
```

这让 negative/positive 都更可解释：

```text
如果 3.5b 赢:
  residual action modeling 本身有用；
  3.5a 的失败更可能来自 joint training / flow objective interference。

如果 3.5b 也输:
  bottleneck 更可能不是 decoder capacity；
  应该回到 upstream transition/event candidate quality 或 sample-prior quality。
```

## Attribution Ledger

3.5b 仍然保留 Gate 3.4 的归因账本。MSE 越低越好，所以 gain 都按
`control - target` 计算：

```text
decoder gain  = frozen temporal_actions MSE - adapter_actions MSE
prior gain    = context-only/no-prior adapter MSE - full aligned adapter MSE
metadata gain = shuffled/rank-prob-only adapter MSE - full aligned adapter MSE
diversity gain = mean_repeated adapter MSE - full aligned adapter MSE
```

先只跑 full aligned seeds 7/17。只有当它先 beat Gate 3.4 temporal baseline
`0.034262`，才扩展 controls：

```text
full event/rank/prob samples
shuffled event metadata
rank/prob-only
mean_repeated
context-only/no-prior
eval-time mean collapse
permutation sanity
batch mismatch
```

如果 richer adapter 让 context-only 也同样好，那是 decoder capacity 吃掉了
贡献；如果只有 aligned GeoMoCo-WM samples + aligned event metadata 好，归因
才站得住。

