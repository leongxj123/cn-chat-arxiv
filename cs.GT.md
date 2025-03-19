# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Last-iterate Convergence Algorithms in Solving Games.](http://arxiv.org/abs/2308.11256) | 该论文研究了求解博弈中高效收敛算法的问题，通过分析乐观梯度下降上升（OGDA）和乐观乘法权重更新（OMWU）算法，以及基于奖励转化（RT）框架的算法，提出了解决这些问题的方法。 |

# 详细

[^1]: 在求解博弈中的高效收敛算法

    Efficient Last-iterate Convergence Algorithms in Solving Games. (arXiv:2308.11256v1 [cs.GT])

    [http://arxiv.org/abs/2308.11256](http://arxiv.org/abs/2308.11256)

    该论文研究了求解博弈中高效收敛算法的问题，通过分析乐观梯度下降上升（OGDA）和乐观乘法权重更新（OMWU）算法，以及基于奖励转化（RT）框架的算法，提出了解决这些问题的方法。

    

    无悔算法在学习两人零和标准型游戏和扩展型游戏的纳什均衡中很受欢迎。最近的许多研究考虑了最后一次迭代收敛的无悔算法。其中，最有名的两个算法是乐观梯度下降上升（OGDA）和乐观乘法权重更新（OMWU）。然而，OGDA的每次迭代复杂度很高。OMWU具有较低的每次迭代复杂度，但实验性能较差，并且它的收敛仅在纳什均衡唯一时成立。最近的研究提出了一种基于奖励转化（RT）框架用于MWU，它消除了唯一性条件，并且在与OMWU相同迭代次数的情况下实现了有竞争力的性能。不幸的是，基于RT的算法在相同迭代次数下表现不如OGDA，并且它们的收敛保证基于连续时间反馈假设，这在大多数情况下不成立。为了解决这些问题，我们对RT框架进行了更详细的分析。

    No-regret algorithms are popular for learning Nash equilibrium (NE) in two-player zero-sum normal-form games (NFGs) and extensive-form games (EFGs). Many recent works consider the last-iterate convergence no-regret algorithms. Among them, the two most famous algorithms are Optimistic Gradient Descent Ascent (OGDA) and Optimistic Multiplicative Weight Update (OMWU). However, OGDA has high per-iteration complexity. OMWU exhibits a lower per-iteration complexity but poorer empirical performance, and its convergence holds only when NE is unique. Recent works propose a Reward Transformation (RT) framework for MWU, which removes the uniqueness condition and achieves competitive performance with OMWU. Unfortunately, RT-based algorithms perform worse than OGDA under the same number of iterations, and their convergence guarantee is based on the continuous-time feedback assumption, which does not hold in most scenarios. To address these issues, we provide a closer analysis of the RT framework, w
    

