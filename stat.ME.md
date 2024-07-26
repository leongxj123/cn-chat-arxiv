# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Amortized Variational Inference with Coverage Guarantees.](http://arxiv.org/abs/2305.14275) | 提出了一种称为CANVI的方法，通过构建一致化预测器并使用预测效率进行比较，来提供具有保证的后验近似结果。该方法可以快速计算，易于实现，并且对于候选近似器的设计决策无需关注。此外，CANVI能够在无似然的情况下使用。 |
| [^2] | [Nonparametric extensions of randomized response for private confidence sets.](http://arxiv.org/abs/2202.08728) | 本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。 |

# 详细

[^1]: 具有覆盖保证的分摊变分推断

    Amortized Variational Inference with Coverage Guarantees. (arXiv:2305.14275v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2305.14275](http://arxiv.org/abs/2305.14275)

    提出了一种称为CANVI的方法，通过构建一致化预测器并使用预测效率进行比较，来提供具有保证的后验近似结果。该方法可以快速计算，易于实现，并且对于候选近似器的设计决策无需关注。此外，CANVI能够在无似然的情况下使用。

    

    分摊变分推断产生了一个后验近似，可以快速计算给定任何新观测。然而，对于这些近似后验的质量，很少有保证。我们提出了一种称为CANVI的一致化分摊神经变分推断的方法，该方法可扩展、易于实现，并提供了保证的边际覆盖。给定一系列候选的分摊后验近似器，CANVI基于每个候选构建一致化预测器，使用预测效率这个度量标准比较预测器，并返回最高效的预测器。CANVI确保所得到的预测器构建的区域以用户指定的概率水平包含真实值。CANVI对候选近似器的制定决策不关心，并且只需要访问前向模型的样本，可以在无似然的情况下使用。我们证明了预测效率的下界。

    Amortized variational inference produces a posterior approximation that can be rapidly computed given any new observation. Unfortunately, there are few guarantees about the quality of these approximate posteriors. We propose Conformalized Amortized Neural Variational Inference (CANVI), a procedure that is scalable, easily implemented, and provides guaranteed marginal coverage. Given a collection of candidate amortized posterior approximators, CANVI constructs conformalized predictors based on each candidate, compares the predictors using a metric known as predictive efficiency, and returns the most efficient predictor. CANVI ensures that the resulting predictor constructs regions that contain the truth with a user-specified level of probability. CANVI is agnostic to design decisions in formulating the candidate approximators and only requires access to samples from the forward model, permitting its use in likelihood-free settings. We prove lower bounds on the predictive efficiency of t
    
[^2]: 随机响应私有置信集的非参数扩展

    Nonparametric extensions of randomized response for private confidence sets. (arXiv:2202.08728v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.08728](http://arxiv.org/abs/2202.08728)

    本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。

    

    本文提出了一种在局部差分隐私（LDP）约束下执行非参数、非渐进统计推断的方法，用于计算具有均值$\mu^\star$的有界观测$(X_1,\dots,X_n)$的置信区间（CI）和时间均匀置信序列（CS），当只有访问私有数据$(Z_1,\dots,Z_n)$时。为了实现这一点，我们引入了一个非参数的、顺序交互的 Warner 的著名“随机响应”机制的推广，为任意有界随机变量满足 LDP，并提供 CIs 和 CSs，用于访问所得私有化的观测值的均值。例如，我们的结果在固定时间和时间均匀区域都产生了 Hoeffding 不等式的私有模拟。我们将这些 Hoeffding  类型的 CSs 扩展到捕获时间变化（非平稳）的均值，最后说明了如何利用这些方法进行实证。

    This work derives methods for performing nonparametric, nonasymptotic statistical inference for population means under the constraint of local differential privacy (LDP). Given bounded observations $(X_1, \dots, X_n)$ with mean $\mu^\star$ that are privatized into $(Z_1, \dots, Z_n)$, we present confidence intervals (CI) and time-uniform confidence sequences (CS) for $\mu^\star$ when only given access to the privatized data. To achieve this, we introduce a nonparametric and sequentially interactive generalization of Warner's famous ``randomized response'' mechanism, satisfying LDP for arbitrary bounded random variables, and then provide CIs and CSs for their means given access to the resulting privatized observations. For example, our results yield private analogues of Hoeffding's inequality in both fixed-time and time-uniform regimes. We extend these Hoeffding-type CSs to capture time-varying (non-stationary) means, and conclude by illustrating how these methods can be used to conduct
    

