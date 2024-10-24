# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Reinforcement Learning for Global Decision Making in the Presence of Local Agents at Scale](https://arxiv.org/abs/2403.00222) | 该研究提出了SUB-SAMPLE-Q算法，通过对局部代理进行子采样，在指数级别的时间内计算出最佳策略，从而实现了与标准方法相比的指数加速。 |
| [^2] | [Aligning Individual and Collective Objectives in Multi-Agent Cooperation](https://arxiv.org/abs/2402.12416) | 引入Altruistic Gradient Adjustment (AgA)优化方法，利用梯度调整来对齐个体和集体目标，加速收敛到期望解决方案 |

# 详细

[^1]: 存在大规模局部代理的全局决策高效强化学习

    Efficient Reinforcement Learning for Global Decision Making in the Presence of Local Agents at Scale

    [https://arxiv.org/abs/2403.00222](https://arxiv.org/abs/2403.00222)

    该研究提出了SUB-SAMPLE-Q算法，通过对局部代理进行子采样，在指数级别的时间内计算出最佳策略，从而实现了与标准方法相比的指数加速。

    

    我们研究了存在许多局部代理的全局决策的强化学习问题，其中全局决策者做出影响所有局部代理的决策，目标是学习一个最大化全局和局部代理奖励的策略。在这种情况下，可扩展性一直是一个长期存在的挑战，因为状态/动作空间的大小可能会随代理数量指数增长。本文提出了SUB-SAMPLE-Q算法，在此算法中，全局代理对$k\leq n$个局部代理进行子采样以在仅指数于$k$的时间内计算出最佳策略，从而提供了与指数于$n$的标准方法相比的指数加速。我们展示了随着子采样代理数$k$的增加，学到的策略将收敛于顺序为$\tilde{O}(1/\sqrt{k}+\epsilon_{k,m})$的最优策略。

    arXiv:2403.00222v1 Announce Type: new  Abstract: We study reinforcement learning for global decision-making in the presence of many local agents, where the global decision-maker makes decisions affecting all local agents, and the objective is to learn a policy that maximizes the rewards of both the global and the local agents. Such problems find many applications, e.g. demand response, EV charging, queueing, etc. In this setting, scalability has been a long-standing challenge due to the size of the state/action space which can be exponential in the number of agents. This work proposes the SUB-SAMPLE-Q algorithm where the global agent subsamples $k\leq n$ local agents to compute an optimal policy in time that is only exponential in $k$, providing an exponential speedup from standard methods that are exponential in $n$. We show that the learned policy converges to the optimal policy in the order of $\tilde{O}(1/\sqrt{k}+\epsilon_{k,m})$ as the number of sub-sampled agents $k$ increases, 
    
[^2]: 在多智能体合作中对齐个体和集体目标

    Aligning Individual and Collective Objectives in Multi-Agent Cooperation

    [https://arxiv.org/abs/2402.12416](https://arxiv.org/abs/2402.12416)

    引入Altruistic Gradient Adjustment (AgA)优化方法，利用梯度调整来对齐个体和集体目标，加速收敛到期望解决方案

    

    在多智能体学习领域，面临着混合动机合作的挑战，因为个体和集体目标之间存在固有的矛盾。当前在该领域的研究主要集中在将领域知识纳入奖励或引入额外机制来促进合作。然而，许多这些方法存在着手动设计成本和缺乏理论基础的收敛程序解决方案的缺点。为了填补这一空白，我们将混合动机博弈建模为一个可微分的博弈以研究学习动态。我们提出了一种名为Altruistic Gradient Adjustment (AgA)的新优化方法，利用梯度调整来新颖地对齐个体和集体目标。此外，我们提供理论证明，AgA中选择适当的对齐权重可以加速收敛到期望解决方案。

    arXiv:2402.12416v1 Announce Type: cross  Abstract: In the field of multi-agent learning, the challenge of mixed-motive cooperation is pronounced, given the inherent contradictions between individual and collective goals. Current research in this domain primarily focuses on incorporating domain knowledge into rewards or introducing additional mechanisms to foster cooperation. However, many of these methods suffer from the drawbacks of manual design costs and the lack of a theoretical grounding convergence procedure to the solution. To address this gap, we approach the mixed-motive game by modeling it as a differentiable game to study learning dynamics. We introduce a novel optimization method named Altruistic Gradient Adjustment (AgA) that employs gradient adjustments to novelly align individual and collective objectives. Furthermore, we provide theoretical proof that the selection of an appropriate alignment weight in AgA can accelerate convergence towards the desired solutions while e
    

