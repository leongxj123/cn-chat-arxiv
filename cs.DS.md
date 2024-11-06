# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Matroid Optimization through Fast Imprecise Oracles](https://arxiv.org/abs/2402.02774) | 本论文研究了如何通过使用快速但不准确的预测模型来加速拟阵优化问题，并提出了实际算法，这些算法在维持对不同质量的预测模型的鲁棒性的同时，只使用了很少的查询 |
| [^2] | [Optimal Scalarizations for Sublinear Hypervolume Regret.](http://arxiv.org/abs/2307.03288) | 研究了用于亚线性超体积遗憾度量的最优标量化方法，证明了具有均匀随机权重的超体积标量化方法在最小化超体积遗憾方面是最优的，并在多目标随机线性赌博机问题上进行了案例研究。 |

# 详细

[^1]: 通过快速不准确的预测优化加速拟阵问题

    Accelerating Matroid Optimization through Fast Imprecise Oracles

    [https://arxiv.org/abs/2402.02774](https://arxiv.org/abs/2402.02774)

    本论文研究了如何通过使用快速但不准确的预测模型来加速拟阵优化问题，并提出了实际算法，这些算法在维持对不同质量的预测模型的鲁棒性的同时，只使用了很少的查询

    

    查询复杂模型以获得准确信息（例如流量模型、数据库系统、大型机器学习模型）通常需要耗费大量计算资源和较长的响应时间。因此，如果可以用较少的查询强模型解决不准确结果的问题，那么使用能够快速给出不准确结果的较弱模型是有优势的。在计算一个拟阵的最大权重基础的基础问题中，这个问题是许多组合优化问题的一个已知泛化。算法可以使用一个干净的查询拟阵信息的预测模型。我们额外提供了一个快速但脏的预测模型来模拟一个未知的、可能不同的拟阵。我们设计和分析了实际算法，这些算法只使用很少数量的干净查询相对于脏预测模型的质量，同时保持对任意质量差的脏拟阵的强健性，并接近给定问题的经典算法的性能。值得注意的是，我们证明了在许多方面我们的算法是最佳的

    Querying complex models for precise information (e.g. traffic models, database systems, large ML models) often entails intense computations and results in long response times. Thus, weaker models which give imprecise results quickly can be advantageous, provided inaccuracies can be resolved using few queries to a stronger model. In the fundamental problem of computing a maximum-weight basis of a matroid, a well-known generalization of many combinatorial optimization problems, algorithms have access to a clean oracle to query matroid information. We additionally equip algorithms with a fast but dirty oracle modelling an unknown, potentially different matroid. We design and analyze practical algorithms which only use few clean queries w.r.t. the quality of the dirty oracle, while maintaining robustness against arbitrarily poor dirty matroids, approaching the performance of classic algorithms for the given problem. Notably, we prove that our algorithms are, in many respects, best-possible
    
[^2]: 用于亚线性超体积遗憾度量的最优标量化方法

    Optimal Scalarizations for Sublinear Hypervolume Regret. (arXiv:2307.03288v1 [cs.LG])

    [http://arxiv.org/abs/2307.03288](http://arxiv.org/abs/2307.03288)

    研究了用于亚线性超体积遗憾度量的最优标量化方法，证明了具有均匀随机权重的超体积标量化方法在最小化超体积遗憾方面是最优的，并在多目标随机线性赌博机问题上进行了案例研究。

    

    标量化是一种通用的技术，可以应用于任何多目标设置中，将多个目标减少为一个，例如最近在RLHF中用于训练校准人类偏好的奖励模型。然而，一些人对这种经典方法持否定态度，因为已知线性标量化会忽略帕累托前沿的凹区域。为此，我们旨在找到简单的非线性标量化方法，以通过被支配的超体积来探索帕累托前沿上的多样化目标集。我们证明，具有均匀随机权重的超体积标量化令人惊讶地是为了证明最小化超体积遗憾而最优的，实现了 $O(T^{-1/k})$ 的最优亚线性遗憾界，同时匹配的下界表明在渐近情况下没有任何算法能做得更好。作为一个理论案例研究，我们考虑了多目标随机线性赌博机问题，并展示了通过利用超线性遗憾界的超体积标量化方法，

    Scalarization is a general technique that can be deployed in any multiobjective setting to reduce multiple objectives into one, such as recently in RLHF for training reward models that align human preferences. Yet some have dismissed this classical approach because linear scalarizations are known to miss concave regions of the Pareto frontier. To that end, we aim to find simple non-linear scalarizations that can explore a diverse set of $k$ objectives on the Pareto frontier, as measured by the dominated hypervolume. We show that hypervolume scalarizations with uniformly random weights are surprisingly optimal for provably minimizing the hypervolume regret, achieving an optimal sublinear regret bound of $O(T^{-1/k})$, with matching lower bounds that preclude any algorithm from doing better asymptotically. As a theoretical case study, we consider the multiobjective stochastic linear bandits problem and demonstrate that by exploiting the sublinear regret bounds of the hypervolume scalariz
    

