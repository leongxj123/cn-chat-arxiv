# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faster Rates for Switchback Experiments](https://arxiv.org/abs/2312.15574) | 本研究提出了一种更快速的Switchback实验方法，通过使用整个时间块，以 $\sqrt{\log T/T}$ 的速率估计全局平均处理效应。 |

# 详细

[^1]: 更快速的Switchback实验方法

    Faster Rates for Switchback Experiments

    [https://arxiv.org/abs/2312.15574](https://arxiv.org/abs/2312.15574)

    本研究提出了一种更快速的Switchback实验方法，通过使用整个时间块，以 $\sqrt{\log T/T}$ 的速率估计全局平均处理效应。

    

    Switchback实验设计中，一个单独的单元（例如整个系统）在交替的时间块中暴露于一个随机处理，处理并行处理了跨单元和时间干扰问题。Hu和Wager（2022）最近提出了一种截断块起始的处理效应估计器，并在Markov条件下证明了用于估计全局平均处理效应（GATE）的$T^{-1/3}$速率，他们声称这个速率是最优的，并建议将注意力转向不同（且依赖设计）的估计量，以获得更快的速率。对于相同的设计，我们提出了一种替代估计器，使用整个块，并惊人地证明，在相同的假设下，它实际上达到了原始的设计独立GATE估计量的$\sqrt{\log T/T}$的估计速率。

    Switchback experimental design, wherein a single unit (e.g., a whole system) is exposed to a single random treatment for interspersed blocks of time, tackles both cross-unit and temporal interference. Hu and Wager (2022) recently proposed a treatment-effect estimator that truncates the beginnings of blocks and established a $T^{-1/3}$ rate for estimating the global average treatment effect (GATE) in a Markov setting with rapid mixing. They claim this rate is optimal and suggest focusing instead on a different (and design-dependent) estimand so as to enjoy a faster rate. For the same design we propose an alternative estimator that uses the whole block and surprisingly show that it in fact achieves an estimation rate of $\sqrt{\log T/T}$ for the original design-independent GATE estimand under the same assumptions.
    

