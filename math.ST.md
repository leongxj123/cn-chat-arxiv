# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality.](http://arxiv.org/abs/2212.09900) | 本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。 |

# 详细

[^1]: 无交叠策略学习：悲观和广义经验Bernstein不等式

    Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality. (arXiv:2212.09900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.09900](http://arxiv.org/abs/2212.09900)

    本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。

    

    本文研究了离线策略学习，旨在利用先前收集到的观测（来自于固定的或是适应演变的行为策略）来学习给定类别中的最优个性化决策规则。现有的策略学习方法依赖于一个统一交叠假设，即离线数据集中探索所有个性化特征的所有动作的倾向性下界。换句话说，这些方法的性能取决于离线数据集中最坏的倾向性。由于数据收集过程不受控制，在许多情况下，这种假设可能不太现实，特别是当允许行为策略随时间演变并且倾向性减弱时。为此，本文提出了一种新的算法，它优化策略价值的下限置信区间（LCBs）——而不是点估计。LCBs通过量化增强倒数倾向权重的估计不确定性来构建。

    This paper studies offline policy learning, which aims at utilizing observations collected a priori (from either fixed or adaptively evolving behavior policies) to learn the optimal individualized decision rule in a given class. Existing policy learning methods rely on a uniform overlap assumption, i.e., the propensities of exploring all actions for all individual characteristics are lower bounded in the offline dataset. In other words, the performance of these methods depends on the worst-case propensity in the offline dataset. As one has no control over the data collection process, this assumption can be unrealistic in many situations, especially when the behavior policies are allowed to evolve over time with diminishing propensities.  In this paper, we propose a new algorithm that optimizes lower confidence bounds (LCBs) -- instead of point estimates -- of the policy values. The LCBs are constructed by quantifying the estimation uncertainty of the augmented inverse propensity weight
    

