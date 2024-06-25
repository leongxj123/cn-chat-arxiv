# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faster Rates for Switchback Experiments](https://arxiv.org/abs/2312.15574) | 本研究提出了一种更快速的Switchback实验方法，通过使用整个时间块，以 $\sqrt{\log T/T}$ 的速率估计全局平均处理效应。 |
| [^2] | [Communication-Efficient Distributed Estimation and Inference for Cox's Model.](http://arxiv.org/abs/2302.12111) | 我们提出了一种高效的分布式算法，用于在高维稀疏Cox比例风险模型中估计和推断，通过引入一种新的去偏差方法，我们可以产生渐近有效的分布式置信区间，并提供了有效的分布式假设检验。 |

# 详细

[^1]: 更快速的Switchback实验方法

    Faster Rates for Switchback Experiments

    [https://arxiv.org/abs/2312.15574](https://arxiv.org/abs/2312.15574)

    本研究提出了一种更快速的Switchback实验方法，通过使用整个时间块，以 $\sqrt{\log T/T}$ 的速率估计全局平均处理效应。

    

    Switchback实验设计中，一个单独的单元（例如整个系统）在交替的时间块中暴露于一个随机处理，处理并行处理了跨单元和时间干扰问题。Hu和Wager（2022）最近提出了一种截断块起始的处理效应估计器，并在Markov条件下证明了用于估计全局平均处理效应（GATE）的$T^{-1/3}$速率，他们声称这个速率是最优的，并建议将注意力转向不同（且依赖设计）的估计量，以获得更快的速率。对于相同的设计，我们提出了一种替代估计器，使用整个块，并惊人地证明，在相同的假设下，它实际上达到了原始的设计独立GATE估计量的$\sqrt{\log T/T}$的估计速率。

    Switchback experimental design, wherein a single unit (e.g., a whole system) is exposed to a single random treatment for interspersed blocks of time, tackles both cross-unit and temporal interference. Hu and Wager (2022) recently proposed a treatment-effect estimator that truncates the beginnings of blocks and established a $T^{-1/3}$ rate for estimating the global average treatment effect (GATE) in a Markov setting with rapid mixing. They claim this rate is optimal and suggest focusing instead on a different (and design-dependent) estimand so as to enjoy a faster rate. For the same design we propose an alternative estimator that uses the whole block and surprisingly show that it in fact achieves an estimation rate of $\sqrt{\log T/T}$ for the original design-independent GATE estimand under the same assumptions.
    
[^2]: 面向Cox模型的高效通信式分布式估计和推断

    Communication-Efficient Distributed Estimation and Inference for Cox's Model. (arXiv:2302.12111v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.12111](http://arxiv.org/abs/2302.12111)

    我们提出了一种高效的分布式算法，用于在高维稀疏Cox比例风险模型中估计和推断，通过引入一种新的去偏差方法，我们可以产生渐近有效的分布式置信区间，并提供了有效的分布式假设检验。

    

    针对因隐私和所有权问题无法共享个体数据的多中心生物医学研究，我们开发了高维稀疏Cox比例风险模型的通信高效迭代分布式算法用于估计和推断。我们证明了即使进行了相对较少的迭代，我们的估计值在非常温和的条件下可以达到与理想全样本估计值相同的收敛速度。为了构建高维危险回归系数的线性组合的置信区间，我们引入了一种新的去偏差方法，建立了中心极限定理，并提供了一致的方差估计，可以产生渐近有效的分布式置信区间。此外，我们提供了基于装饰分数检验的任意坐标元素的有效和强大的分布式假设检验。我们还允许时间依赖协变量以及被审查的生存时间。在多种数据集上进行了广泛的数字实验，证明了算法的有效性和效率。

    Motivated by multi-center biomedical studies that cannot share individual data due to privacy and ownership concerns, we develop communication-efficient iterative distributed algorithms for estimation and inference in the high-dimensional sparse Cox proportional hazards model. We demonstrate that our estimator, even with a relatively small number of iterations, achieves the same convergence rate as the ideal full-sample estimator under very mild conditions. To construct confidence intervals for linear combinations of high-dimensional hazard regression coefficients, we introduce a novel debiased method, establish central limit theorems, and provide consistent variance estimators that yield asymptotically valid distributed confidence intervals. In addition, we provide valid and powerful distributed hypothesis tests for any coordinate element based on a decorrelated score test. We allow time-dependent covariates as well as censored survival times. Extensive numerical experiments on both s
    

