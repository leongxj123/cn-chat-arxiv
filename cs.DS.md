# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerated Algorithms for Constrained Nonconvex-Nonconcave Min-Max Optimization and Comonotone Inclusion](https://arxiv.org/abs/2206.05248) | 本论文提出了针对约束共单调极小-极大优化和共单调包含问题的加速算法，扩展了现有算法并实现了较优的收敛速率，同时证明了算法的收敛性。 |
| [^2] | [The Power of Populations in Decentralized Learning Dynamics.](http://arxiv.org/abs/2306.08670) | 本文研究了分散式学习动力学中个体群体的力量。我们介绍了一种分散式的多臂赌博机设置，并分析了几个针对此任务的分散式动力学家族。我们展示了这些动力学与一类“零和”乘法权重更新算法的联系，并开发了一个通用框架来分析这些协议的群体级遗憾。在广泛的参数范围下，我们得到了次线性的遗憾界限。 |

# 详细

[^1]: 加速算法用于约束非凸-非凹极小-极大优化和共单调包含

    Accelerated Algorithms for Constrained Nonconvex-Nonconcave Min-Max Optimization and Comonotone Inclusion

    [https://arxiv.org/abs/2206.05248](https://arxiv.org/abs/2206.05248)

    本论文提出了针对约束共单调极小-极大优化和共单调包含问题的加速算法，扩展了现有算法并实现了较优的收敛速率，同时证明了算法的收敛性。

    

    我们研究了约束共单调极小-极大优化，一类结构化的非凸-非凹极小-极大优化问题以及它们对共单调包含的推广。在我们的第一个贡献中，我们将最初由Yoon和Ryu（2021）提出的无约束极小-极大优化的Extra Anchored Gradient（EAG）算法扩展到约束共单调极小-极大优化和共单调包含问题，并实现了所有一阶方法中的最优收敛速率$O\left(\frac{1}{T}\right)$。此外，我们证明了算法的迭代收敛到解集中的一个点。在我们的第二个贡献中，我们将由Lee和Kim（2021）开发的快速额外梯度（FEG）算法扩展到约束共单调极小-极大优化和共单调包含，并实现了相同的$O\left(\frac{1}{T}\right)$收敛速率。这个速率适用于文献中研究过的最广泛的共单调包含问题集合。我们的分析基于s的内容。

    We study constrained comonotone min-max optimization, a structured class of nonconvex-nonconcave min-max optimization problems, and their generalization to comonotone inclusion. In our first contribution, we extend the Extra Anchored Gradient (EAG) algorithm, originally proposed by Yoon and Ryu (2021) for unconstrained min-max optimization, to constrained comonotone min-max optimization and comonotone inclusion, achieving an optimal convergence rate of $O\left(\frac{1}{T}\right)$ among all first-order methods. Additionally, we prove that the algorithm's iterations converge to a point in the solution set. In our second contribution, we extend the Fast Extra Gradient (FEG) algorithm, as developed by Lee and Kim (2021), to constrained comonotone min-max optimization and comonotone inclusion, achieving the same $O\left(\frac{1}{T}\right)$ convergence rate. This rate is applicable to the broadest set of comonotone inclusion problems yet studied in the literature. Our analyses are based on s
    
[^2]: 分散式学习动力学中个体群体的力量

    The Power of Populations in Decentralized Learning Dynamics. (arXiv:2306.08670v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.08670](http://arxiv.org/abs/2306.08670)

    本文研究了分散式学习动力学中个体群体的力量。我们介绍了一种分散式的多臂赌博机设置，并分析了几个针对此任务的分散式动力学家族。我们展示了这些动力学与一类“零和”乘法权重更新算法的联系，并开发了一个通用框架来分析这些协议的群体级遗憾。在广泛的参数范围下，我们得到了次线性的遗憾界限。

    

    我们研究了一种分散式多臂赌博机设置，在一个由$n$个受内存限制的节点组成的种群中，采用了谣言模型：每轮，每个节点本地采用$m$个臂之一，观察从臂的（对抗选择的）分布中抽取的奖励，然后与随机抽取的邻居进行通信，交换信息，以确定下一轮的策略。我们介绍并分析了几个针对此任务的分散式动力学家族：每个节点的决策完全是局部的，只依赖于其最新获得的奖励以及它抽样的邻居的奖励。我们展示了这些分散式动力学的全局演化与特定类型的“零和”乘法权重更新算法之间的联系，并且开发了一个分析这些自然协议的群体级遗憾的通用框架。利用这个框架，我们在广泛的参数范围（即，种群的大小和nu的大小）下推导了次线性遗憾界限。

    We study a distributed multi-armed bandit setting among a population of $n$ memory-constrained nodes in the gossip model: at each round, every node locally adopts one of $m$ arms, observes a reward drawn from the arm's (adversarially chosen) distribution, and then communicates with a randomly sampled neighbor, exchanging information to determine its policy in the next round. We introduce and analyze several families of dynamics for this task that are decentralized: each node's decision is entirely local and depends only on its most recently obtained reward and that of the neighbor it sampled. We show a connection between the global evolution of these decentralized dynamics with a certain class of "zero-sum" multiplicative weights update algorithms, and we develop a general framework for analyzing the population-level regret of these natural protocols. Using this framework, we derive sublinear regret bounds under a wide range of parameter regimes (i.e., the size of the population and nu
    

