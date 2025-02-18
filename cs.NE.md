# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Population Update Can Provably Be Helpful in Multi-Objective Evolutionary Algorithms](https://arxiv.org/abs/2306.02611) | 本研究通过理论分析证明，在多目标进化算法中采用随机种群更新机制可以显著降低算法的运行时间，从而提高问题的求解效率。 |

# 详细

[^1]: 随机种群更新在多目标进化算法中可以被证明是有帮助的

    Stochastic Population Update Can Provably Be Helpful in Multi-Objective Evolutionary Algorithms

    [https://arxiv.org/abs/2306.02611](https://arxiv.org/abs/2306.02611)

    本研究通过理论分析证明，在多目标进化算法中采用随机种群更新机制可以显著降低算法的运行时间，从而提高问题的求解效率。

    

    进化算法（EAs）因其基于种群的搜索特性，已被广泛且成功地应用于解决多目标优化问题。种群更新是多目标进化算法（MOEAs）中的关键组成部分，通常以贪婪、确定性的方式进行。也就是说，下一代种群是通过从当前种群和新生成的解中选择最优解形成的（无论使用的选择标准是Pareto支配、拥挤度还是指标等）。本文对这种做法提出了质疑。我们从理论上证明了随机种群更新对于MOEAs的搜索是有益的。具体地，我们证明了将确定性种群更新机制替换为随机机制，可以指数级减少两个已经被广泛接受的MOEAs（SMS-EMOA和NSGA-II）在解决两个双目标问题（OneJumpZeroJump和双目标RealRoyalRoad）上的预计运行时间。此外，还进行了实证研究。

    Evolutionary algorithms (EAs) have been widely and successfully applied to solve multi-objective optimization problems, due to their nature of population-based search. Population update, a key component in multi-objective EAs (MOEAs), is usually performed in a greedy, deterministic manner. That is, the next-generation population is formed by selecting the best solutions from the current population and newly-generated solutions (irrespective of the selection criteria used such as Pareto dominance, crowdedness and indicators). In this paper, we question this practice. We analytically present that stochastic population update can be beneficial for the search of MOEAs. Specifically, we prove that the expected running time of two well-established MOEAs, SMS-EMOA and NSGA-II, for solving two bi-objective problems, OneJumpZeroJump and bi-objective RealRoyalRoad, can be exponentially decreased if replacing its deterministic population update mechanism by a stochastic one. Empirical studies als
    

