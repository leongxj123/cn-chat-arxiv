# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk.](http://arxiv.org/abs/2304.04341) | 该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。 |

# 详细

[^1]: 随机赌博机中的遗憾分布：期望和尾部风险之间的最优权衡

    Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk. (arXiv:2304.04341v1 [stat.ML])

    [http://arxiv.org/abs/2304.04341](http://arxiv.org/abs/2304.04341)

    该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。

    

    本文研究了随机多臂赌博问题中，遗憾分布的期望和尾部风险之间的权衡问题。我们完全刻画了策略设计中三个期望性质之间的相互作用：最坏情况下的最优性，实例相关的一致性和轻尾风险。我们展示了期望遗憾的顺序如何影响遗憾尾部概率的衰减率，同时包括了最坏情况和实例相关的情况。我们提出了一种新的策略，以表征对于任何遗憾阈值的最优遗憾尾部概率。具体地，对于任何给定的$\alpha \in [1/2, 1)$和$\beta \in [0, \alpha]$，我们的策略可以实现平均期望遗憾$\tilde O(T^\alpha)$的最坏情况下$\alpha$-最优和期望遗憾$\tilde O(T^\beta)$的实例相关的$\beta$-一致性，并且享有一定的概率可以避免$\tilde O(T^\delta)$的遗憾($\delta \geq \alpha$在最坏情况下和$\delta \geq \beta$在实例相关的情况下)。

    We study the trade-off between expectation and tail risk for regret distribution in the stochastic multi-armed bandit problem. We fully characterize the interplay among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to characterize the optimal regret tail probability for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, \alpha]$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ (we call it $\alpha$-optimal) and an instance-dependent expected regret of $\tilde O(T^\beta)$ (we call it $\beta$-consistent), while enjoys a probability of incurring an $\tilde O(T^\delta)$ regret ($\delta\geq\alpha$ in the worst-case scenario and $\delta\geq\beta$ in the instance-dependent s
    

