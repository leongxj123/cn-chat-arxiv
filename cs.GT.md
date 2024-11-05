# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mixed-Strategy Nash Equilibrium for Crowd Navigation](https://arxiv.org/abs/2403.01537) | 通过简单的迭代贝叶斯更新方案和基于数据驱动的框架，我们证明了混合策略纳什均衡模型为人群导航提供了实时且可扩展的决策制定方法。 |
| [^2] | [DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation.](http://arxiv.org/abs/2306.02071) | 本论文提出了一种称为DU-Shapley的方法，用于更有效地计算Shapley值，以实现机器学习中的数据集价值评估。 |
| [^3] | [Proportionally Representative Clustering.](http://arxiv.org/abs/2304.13917) | 本文提出了一个新的公平性准则——比例代表性公平性（PRF），并设计了有效的算法满足该准则。 |

# 详细

[^1]: 混合策略纳什均衡用于人群导航

    Mixed-Strategy Nash Equilibrium for Crowd Navigation

    [https://arxiv.org/abs/2403.01537](https://arxiv.org/abs/2403.01537)

    通过简单的迭代贝叶斯更新方案和基于数据驱动的框架，我们证明了混合策略纳什均衡模型为人群导航提供了实时且可扩展的决策制定方法。

    

    我们解决了针对人群导航找到混合策略纳什均衡的问题。混合策略纳什均衡为机器人提供了一个严谨的模型，使其能够预测人群中不确定但合作的人类行为，但计算成本通常太高，无法进行可扩展和实时的决策制定。在这里，我们证明了一个简单的迭代贝叶斯更新方案收敛于混合策略社交导航游戏的纳什均衡。此外，我们提出了一个基于数据驱动的框架，通过将代理策略初始化为从人类数据集学习的高斯过程，来构建该游戏。基于所提出的混合策略纳什均衡模型，我们开发了一个基于采样的人群导航框架，可以集成到现有导航方法中，并可在笔记本电脑 CPU 上实时运行。我们通过模拟环境和真实世界的非结构化环境中人类数据集对我们的框架进行了评估。

    arXiv:2403.01537v1 Announce Type: cross  Abstract: We address the problem of finding mixed-strategy Nash equilibrium for crowd navigation. Mixed-strategy Nash equilibrium provides a rigorous model for the robot to anticipate uncertain yet cooperative human behavior in crowds, but the computation cost is often too high for scalable and real-time decision-making. Here we prove that a simple iterative Bayesian updating scheme converges to the Nash equilibrium of a mixed-strategy social navigation game. Furthermore, we propose a data-driven framework to construct the game by initializing agent strategies as Gaussian processes learned from human datasets. Based on the proposed mixed-strategy Nash equilibrium model, we develop a sampling-based crowd navigation framework that can be integrated into existing navigation methods and runs in real-time on a laptop CPU. We evaluate our framework in both simulated environments and real-world human datasets in unstructured environments. Our framework
    
[^2]: DU-Shapley: 一种有效的数据集价值评估的Shapley值代理

    DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation. (arXiv:2306.02071v1 [cs.AI])

    [http://arxiv.org/abs/2306.02071](http://arxiv.org/abs/2306.02071)

    本论文提出了一种称为DU-Shapley的方法，用于更有效地计算Shapley值，以实现机器学习中的数据集价值评估。

    

    许多机器学习问题需要进行数据集评估，即量化将一个单独的数据集与其他数据集聚合的增量收益，以某些相关预定义公用事业为基础。最近，Shapley值被提出作为实现这一目标的一种基本工具，因为它具有形式公理证明。由于其计算通常需要指数时间，因此考虑基于Monte Carlo积分的标准近似策略。然而，在某些情况下，这种通用近似方法仍然昂贵。本文利用数据集评估问题的结构知识，设计了更有效的Shapley值估计器。我们提出了一种新的Shapley值近似，称为离散均匀Shapley (DU-Shapley)，其表达为期望值

    Many machine learning problems require performing dataset valuation, i.e. to quantify the incremental gain, to some relevant pre-defined utility, of aggregating an individual dataset to others. As seminal examples, dataset valuation has been leveraged in collaborative and federated learning to create incentives for data sharing across several data owners. The Shapley value has recently been proposed as a principled tool to achieve this goal due to formal axiomatic justification. Since its computation often requires exponential time, standard approximation strategies based on Monte Carlo integration have been considered. Such generic approximation methods, however, remain expensive in some cases. In this paper, we exploit the knowledge about the structure of the dataset valuation problem to devise more efficient Shapley value estimators. We propose a novel approximation of the Shapley value, referred to as discrete uniform Shapley (DU-Shapley) which is expressed as an expectation under 
    
[^3]: 比例代表性聚类

    Proportionally Representative Clustering. (arXiv:2304.13917v1 [cs.LG])

    [http://arxiv.org/abs/2304.13917](http://arxiv.org/abs/2304.13917)

    本文提出了一个新的公平性准则——比例代表性公平性（PRF），并设计了有效的算法满足该准则。

    

    近年来，机器学习领域对公平概念的形式化表述越来越受关注。本文关注于聚类，是无监督机器学习中最基础的任务之一。我们提出了一个新的公平性准则——比例代表性公平性（PRF），我们认为该概念以一种更有说服力的方式达到了文献中几个现存概念的理由。但现有的公平聚类算法不能满足我们的公平性概念。我们设计了高效的算法，以满足无约束聚类和离散聚类问题的PRF。

    In recent years, there has been a surge in effort to formalize notions of fairness in machine learning. We focus on clustering -- one of the fundamental tasks in unsupervised machine learning. We propose a new axiom that captures proportional representation fairness (PRF). We make a case that the concept achieves the raison d'{\^{e}}tre of several existing concepts in the literature in an arguably more convincing manner. Our fairness concept is not satisfied by existing fair clustering algorithms. We design efficient algorithms to achieve PRF both for unconstrained and discrete clustering problems.
    

