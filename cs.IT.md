# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model.](http://arxiv.org/abs/2305.16589) | 本文研究了强化学习中的模型鲁棒性以缩小模拟与真实差距，提出了一个名为“分布鲁棒值迭代”的基于模型的方法，可以优化最坏情况下的表现。 |

# 详细

[^1]: 具有生成模型的强化学习中分布鲁棒性的可疑价格

    The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model. (arXiv:2305.16589v1 [cs.LG])

    [http://arxiv.org/abs/2305.16589](http://arxiv.org/abs/2305.16589)

    本文研究了强化学习中的模型鲁棒性以缩小模拟与真实差距，提出了一个名为“分布鲁棒值迭代”的基于模型的方法，可以优化最坏情况下的表现。

    

    本文研究了强化学习中的模型鲁棒性，以减少在实践中的模拟与真实差距。我们采用分布鲁棒马尔可夫决策过程（RMDPs）框架，旨在学习一个策略，在部署环境落在预定的不确定性集合内时，优化最坏情况下的表现。尽管最近有了一些努力，但RMDPs的样本复杂度仍然没有得到解决，无论使用的不确定性集合是什么。不清楚分布鲁棒性与标准强化学习相比是否具有统计学上的影响。假设有一个生成模型，根据名义MDP绘制样本，我们将描述RMDPs的样本复杂度，当由总变差（TV）距离或$\chi^2$分歧指定不确定性集合时。在这里研究的算法是一种基于模型的方法，称为分布鲁棒值迭代，证明了它在整个范围内都是近乎最优的。

    This paper investigates model robustness in reinforcement learning (RL) to reduce the sim-to-real gap in practice. We adopt the framework of distributionally robust Markov decision processes (RMDPs), aimed at learning a policy that optimizes the worst-case performance when the deployed environment falls within a prescribed uncertainty set around the nominal MDP. Despite recent efforts, the sample complexity of RMDPs remained mostly unsettled regardless of the uncertainty set in use. It was unclear if distributional robustness bears any statistical consequences when benchmarked against standard RL.  Assuming access to a generative model that draws samples based on the nominal MDP, we characterize the sample complexity of RMDPs when the uncertainty set is specified via either the total variation (TV) distance or $\chi^2$ divergence. The algorithm studied here is a model-based method called {\em distributionally robust value iteration}, which is shown to be near-optimal for the full range
    

