# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimate-Then-Optimize Versus Integrated-Estimation-Optimization: A Stochastic Dominance Perspective.](http://arxiv.org/abs/2304.06833) | 本文提出，当模型类足够丰富以涵盖真实情况时，非线性问题的“先估计再优化”方法优于集成方法，包括优化间隙的渐进优势的均值，所有其他时刻和整个渐进分布。 |

# 详细

[^1]: 评估-优化方法与集成评估优化法：基于随机优势的观点

    Estimate-Then-Optimize Versus Integrated-Estimation-Optimization: A Stochastic Dominance Perspective. (arXiv:2304.06833v1 [stat.ML])

    [http://arxiv.org/abs/2304.06833](http://arxiv.org/abs/2304.06833)

    本文提出，当模型类足够丰富以涵盖真实情况时，非线性问题的“先估计再优化”方法优于集成方法，包括优化间隙的渐进优势的均值，所有其他时刻和整个渐进分布。

    

    在数据驱动的随机优化中，除了需要优化任务，还需要从数据中估计潜在分布的模型参数。最近的文献表明，通过选择导致最佳经验目标性能的模型参数，可以集成估计和优化过程。当模型被错误地指定时，这种集成方法可以很容易地显示出优于简单的“先估计再优化”的方法。本文认为，在模型类足够丰富以涵盖真实情况的情况下，对于非线性问题，两种方法之间的性能排序在强烈的意义下被颠倒。在受限条件和当上下文特征可用时，类似的结果也成立。

    In data-driven stochastic optimization, model parameters of the underlying distribution need to be estimated from data in addition to the optimization task. Recent literature suggests the integration of the estimation and optimization processes, by selecting model parameters that lead to the best empirical objective performance. Such an integrated approach can be readily shown to outperform simple ``estimate then optimize" when the model is misspecified. In this paper, we argue that when the model class is rich enough to cover the ground truth, the performance ordering between the two approaches is reversed for nonlinear problems in a strong sense. Simple ``estimate then optimize" outperforms the integrated approach in terms of stochastic dominance of the asymptotic optimality gap, i,e, the mean, all other moments, and the entire asymptotic distribution of the optimality gap is always better. Analogous results also hold under constrained settings and when contextual features are availa
    

