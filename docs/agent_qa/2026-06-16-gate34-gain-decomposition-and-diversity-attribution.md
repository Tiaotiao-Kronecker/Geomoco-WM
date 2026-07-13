# Gate 3.4 Gain Decomposition And Diversity Attribution

## Question

Gate 3.4 的 joint temporal action-sequence decoder 为什么不能只看最终
MSE？为什么要分成 decoder gain、prior gain、metadata gain、diversity gain？

## Short Answer

因为 Gate 3.4 同时改了两个东西：

```text
1. decoder 变强了一点；
2. decoder 仍然消费 GeoMoCo-WM predicted event-mixture samples。
```

如果只看最终 MSE 变好，无法判断贡献来自哪里：

```text
是 temporal decoder 自己学成了更强 BC？
是 GeoMoCo-WM motion prior 仍然有用？
是 event metadata 对齐有用？
还是 K 个 future samples 的多样性真的被用起来了？
```

所以要把 gain 拆开。

## Sign Convention

MSE 越低越好，所以所有 gain 都应该按下面方式计算：

```text
gain = baseline_or_control_MSE - target_MSE
```

gain 为正，说明 target 更好。

## Decoder Gain

问题：

```text
同一个 checkpoint、同一个输入里，temporal_actions 是否比 base actions 好？
```

公式：

```text
decoder gain = base MSE - temporal_action MSE
```

Gate 3.4 full aligned repeated-eval mean:

```text
base overall:     0.034303
temporal overall: 0.034262
decoder gain:     +0.000041

base transition:     0.131697
temporal transition: 0.131311
decoder gain:        +0.000386
```

解释：

```text
joint temporal decoder 有一点用，但 gain 很小。
```

它说明 Gate 3.3 gripper-only additive residual 的失败不等于所有 temporal
decoder 都无效；但它还不是一个强架构胜利。

## Prior Gain

问题：

```text
这个结果是不是只靠 richer decoder 自己学出来的？
还是确实需要 GeoMoCo-WM motion-prior samples？
```

对比：

```text
full aligned temporal
vs
context-only/no-prior temporal
```

公式：

```text
prior gain = context-only MSE - full aligned MSE
```

Gate 3.4 repeated-eval mean:

```text
context-only overall: 0.036642
full aligned overall: 0.034262
prior gain:           +0.002380

context-only transition: 0.136922
full aligned transition: 0.131311
prior gain:              +0.005611
```

解释：

```text
same-capacity decoder alone 解释不了 full aligned 的结果。
GeoMoCo-WM prior 仍然有实质贡献。
```

## Metadata Gain

问题：

```text
motion samples 有用，是不是因为 event metadata 对齐有用？
```

Controls:

```text
shuffled event/rank/prob: motion samples 还在，但 event identity 错配；
rank/prob-only: rank/prob 还在，但 event identity 去掉。
```

公式：

```text
metadata gain = metadata_control_MSE - full aligned MSE
```

Gate 3.4 repeated-eval mean:

```text
shuffled overall: 0.035529
full overall:     0.034262
gain:             +0.001267

rank/prob overall: 0.035875
full overall:      0.034262
gain:              +0.001613

shuffled transition: 0.135399
full transition:     0.131311
gain:                +0.004088

rank/prob transition: 0.135893
full transition:      0.131311
gain:                 +0.004582
```

解释：

```text
aligned event identity + event rank/prob 仍然重要。
Gate 3.4 不是裸 action decoder 在学 BC。
```

## Diversity Gain

问题：

```text
模型到底有没有用到 K 个 motion samples 的多样性？
还是只用了这些 samples 的均值和 metadata？
```

Control:

```text
mean_repeated
```

`mean_repeated` 把原始 K 个 motion samples:

```text
[sample1, sample2, ..., sample16]
```

替换成：

```text
[mean, mean, ..., mean]
```

event metadata 和 decoder capacity 仍然保留，但 motion sample diversity 被抹掉。

公式：

```text
diversity gain = mean_repeated MSE - full aligned MSE
```

Gate 3.4 repeated-eval mean:

```text
mean_repeated overall: 0.034414
full aligned overall:  0.034262
diversity gain:        +0.000152

mean_repeated transition: 0.132199
full aligned transition:  0.131311
diversity gain:           +0.000888
```

解释：

```text
full aligned 确实比 mean_repeated 好，但差距很小。
因此不能强 claim 当前 decoder 充分利用了 K-sample diversity。
```

更稳的结论是：

```text
Gate 3.4 使用了 aligned event metadata 和 motion-prior mean/sample structure；
但 K 个候选 future 的多样性贡献还不够强。
```

## Why This Determines The Next Step

如果现在直接上更强 flow/diffusion decoder，数字可能变好，但 attribution
会更难：

```text
是 GeoMoCo-WM samples 有用？
还是 decoder 变强以后自己学成 temporal BC policy？
```

所以更干净的下一步是 Gate 3.4b:

```text
先诊断 full aligned 到底在哪些窗口、以什么机制赢过 mean_repeated；
再决定是否做 set-wise temporal/regret/rank supervision；
最后才考虑小型 flow/diffusion residual decoder。
```

Gate 3.4b 的目标不是泛泛降低 MSE，而是让下面这个量更清楚：

```text
diversity gain = mean_repeated MSE - full aligned MSE
```

如果 diversity gain 在 transition / high-diversity / high-oracle-gap 窗口上
明显变大，说明 GeoMoCo-WM 的 K-sample future distribution 真的被用起来了。
如果没有，说明 bottleneck 不是 decoder 表达力，而是 sample selection /
set-wise supervision / prior quality。
