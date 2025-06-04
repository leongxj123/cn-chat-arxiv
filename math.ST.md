# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maximum Likelihood Estimation on Stochastic Blockmodels for Directed Graph Clustering](https://arxiv.org/abs/2403.19516) | 本文研究了有向图聚类问题，提出了一种基于有向随机块模型的最大似然估计方法，并引入了两种高效且可解释的有向聚类算法。 |
| [^2] | [Two Phases of Scaling Laws for Nearest Neighbor Classifiers.](http://arxiv.org/abs/2308.08247) | 最近邻分类器的缩放律可分为两个阶段：第一阶段中，泛化误差多项式地依赖于数据维度并迅速减小；第二阶段中，误差指数地依赖于数据维度并缓慢减小。这表明最近邻分类器在数据分布良好时可以实现泛化误差多项式地依赖于数据维度，而不是指数地依赖于数据维度。 |
| [^3] | [Accelerate Langevin Sampling with Birth-Death process and Exploration Component.](http://arxiv.org/abs/2305.05529) | 该论文提出了一种新的采样方法，在探索新模式和传递有用信息的过程中利用了Birth-Death过程和探索组件，具有高效和指数渐近收敛等优点。 |

# 详细

[^1]: 针对有向图聚类问题的随机块模型最大似然估计

    Maximum Likelihood Estimation on Stochastic Blockmodels for Directed Graph Clustering

    [https://arxiv.org/abs/2403.19516](https://arxiv.org/abs/2403.19516)

    本文研究了有向图聚类问题，提出了一种基于有向随机块模型的最大似然估计方法，并引入了两种高效且可解释的有向聚类算法。

    

    本文通过统计学的视角研究了有向图聚类问题，将聚类问题建模为有向随机块模型（DSBM）中潜在社区的估计。我们对DSBM进行最大似然估计（MLE），从而确定给定观察到的图结构时最可能的社区分配。除了统计观点外，我们进一步建立了这种MLE公式与一种新颖的流优化启发式之间的等价性，该启发式同时考虑了两个重要的有向图统计量：边密度和边方向。基于这种有向聚类的新公式，我们引入了两种高效且可解释的有向聚类算法，分别是谱聚类算法和基于半定规划的聚类算法。我们为谱聚类算法的错误聚类顶点数提供了一个理论上界。

    arXiv:2403.19516v1 Announce Type: cross  Abstract: This paper studies the directed graph clustering problem through the lens of statistics, where we formulate clustering as estimating underlying communities in the directed stochastic block model (DSBM). We conduct the maximum likelihood estimation (MLE) on the DSBM and thereby ascertain the most probable community assignment given the observed graph structure. In addition to the statistical point of view, we further establish the equivalence between this MLE formulation and a novel flow optimization heuristic, which jointly considers two important directed graph statistics: edge density and edge orientation. Building on this new formulation of directed clustering, we introduce two efficient and interpretable directed clustering algorithms, a spectral clustering algorithm and a semidefinite programming based clustering algorithm. We provide a theoretical upper bound on the number of misclustered vertices of the spectral clustering algor
    
[^2]: 最近邻分类器的两个阶段的缩放律

    Two Phases of Scaling Laws for Nearest Neighbor Classifiers. (arXiv:2308.08247v1 [stat.ML])

    [http://arxiv.org/abs/2308.08247](http://arxiv.org/abs/2308.08247)

    最近邻分类器的缩放律可分为两个阶段：第一阶段中，泛化误差多项式地依赖于数据维度并迅速减小；第二阶段中，误差指数地依赖于数据维度并缓慢减小。这表明最近邻分类器在数据分布良好时可以实现泛化误差多项式地依赖于数据维度，而不是指数地依赖于数据维度。

    

    缩放律是指当训练数据数量增加时，模型的测试性能会提高的观察结果。快速的缩放律意味着通过增加数据和模型大小就能解决机器学习问题。然而，在许多情况下，增加更多数据的好处可能是微不足道的。在本研究中，我们研究了最近邻分类器的缩放律。我们发现缩放律可能有两个阶段：在第一阶段，泛化误差多项式地依赖于数据维度并且快速减小；而在第二阶段，误差指数地依赖于数据维度并且减小得慢。我们的分析突显了数据分布在决定泛化误差中的复杂性。当数据分布良好时，我们的结果表明最近邻分类器可以实现泛化误差多项式地依赖于数据维度，而不是指数地依赖于数据维度。

    A scaling law refers to the observation that the test performance of a model improves as the number of training data increases. A fast scaling law implies that one can solve machine learning problems by simply boosting the data and the model sizes. Yet, in many cases, the benefit of adding more data can be negligible. In this work, we study the rate of scaling laws of nearest neighbor classifiers. We show that a scaling law can have two phases: in the first phase, the generalization error depends polynomially on the data dimension and decreases fast; whereas in the second phase, the error depends exponentially on the data dimension and decreases slowly. Our analysis highlights the complexity of the data distribution in determining the generalization error. When the data distributes benignly, our result suggests that nearest neighbor classifier can achieve a generalization error that depends polynomially, instead of exponentially, on the data dimension.
    
[^3]: 利用Birth-Death 过程和探索组件加速Langevin采样

    Accelerate Langevin Sampling with Birth-Death process and Exploration Component. (arXiv:2305.05529v1 [stat.CO])

    [http://arxiv.org/abs/2305.05529](http://arxiv.org/abs/2305.05529)

    该论文提出了一种新的采样方法，在探索新模式和传递有用信息的过程中利用了Birth-Death过程和探索组件，具有高效和指数渐近收敛等优点。

    

    在计算科学和工程中，采样已知概率分布是一项基本任务。针对多峰性，我们提出了一种新的采样方法，利用了Birth-Death过程和探索组件。该方法的主要思想是“三思而后行”。我们保留两组采样器，一组在较高温度下，一组在原始温度下。前者作为探索新模式和将有用信息传递给后者的先驱，后者在接收信息后对目标分布进行采样。我们推导了均场极限，并展示了探索过程如何决定采样效率。此外，在温和假设下，我们证明了指数渐近收敛。最后，我们对以前文献中的实验进行了测试，并将我们的方法与以前的方法进行了比较。

    Sampling a probability distribution with known likelihood is a fundamental task in computational science and engineering. Aiming at multimodality, we propose a new sampling method that takes advantage of both birth-death process and exploration component. The main idea of this method is \textit{look before you leap}. We keep two sets of samplers, one at warmer temperature and one at original temperature. The former one serves as pioneer in exploring new modes and passing useful information to the other, while the latter one samples the target distribution after receiving the information. We derive a mean-field limit and show how the exploration process determines sampling efficiency. Moreover, we prove exponential asymptotic convergence under mild assumption. Finally, we test on experiments from previous literature and compared our methodology to previous ones.
    

