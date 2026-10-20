# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ensemble-Based Annealed Importance Sampling.](http://arxiv.org/abs/2401.15645) | 本文提出了一种基于集合的退火重要性抽样算法，通过结合人口蒙特卡洛方法来提高抽样效率，并利用集合的相互作用促进未发现模态的探索。 |

# 详细

[^1]: 基于集合的退火重要性抽样

    Ensemble-Based Annealed Importance Sampling. (arXiv:2401.15645v1 [stat.CO])

    [http://arxiv.org/abs/2401.15645](http://arxiv.org/abs/2401.15645)

    本文提出了一种基于集合的退火重要性抽样算法，通过结合人口蒙特卡洛方法来提高抽样效率，并利用集合的相互作用促进未发现模态的探索。

    

    从多模态分布中进行抽样是计算科学和统计学中的一个基本且具有挑战性的问题。在各种方法中，一种流行的方法是退火重要性抽样（AIS）。在本文中，我们提出了一个基于集合的AIS版本，通过将其与基于人口的蒙特卡洛方法相结合，以提高其效率。通过跟踪集合而不是单个粒子沿起始分布和目标分布之间的某个延续路径，我们利用集合内的相互作用来促进未发现模态的探索。具体来说，我们的主要思想是利用Snooker算法或进化蒙特卡洛中使用的遗传算法。我们讨论了如何实现所提出的算法，并推导了在连续时间和均场极限下控制集合演化的偏微分方程。我们还测试了所提算法的效率。

    Sampling from a multimodal distribution is a fundamental and challenging problem in computational science and statistics. Among various approaches proposed for this task, one popular method is Annealed Importance Sampling (AIS). In this paper, we propose an ensemble-based version of AIS by combining it with population-based Monte Carlo methods to improve its efficiency. By keeping track of an ensemble instead of a single particle along some continuation path between the starting distribution and the target distribution, we take advantage of the interaction within the ensemble to encourage the exploration of undiscovered modes. Specifically, our main idea is to utilize either the snooker algorithm or the genetic algorithm used in Evolutionary Monte Carlo. We discuss how the proposed algorithm can be implemented and derive a partial differential equation governing the evolution of the ensemble under the continuous time and mean-field limit. We also test the efficiency of the proposed alg
    

