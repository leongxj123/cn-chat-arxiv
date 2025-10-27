# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maximal Inequalities for Empirical Processes under General Mixing Conditions with an Application to Strong Approximations](https://arxiv.org/abs/2402.11394) | 本文提出了针对一般混合随机过程的样本均值上确界的界限，不受混合速度的影响，强调了集中速率和复杂度度量的重要性，并发现了混合速度对集中速率的影响，引入了相变的概念。 |
| [^2] | [Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk.](http://arxiv.org/abs/2304.04341) | 该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。 |

# 详细

[^1]: 基于一般混合条件的经验过程的极值不等式及其在强逼近中的应用

    Maximal Inequalities for Empirical Processes under General Mixing Conditions with an Application to Strong Approximations

    [https://arxiv.org/abs/2402.11394](https://arxiv.org/abs/2402.11394)

    本文提出了针对一般混合随机过程的样本均值上确界的界限，不受混合速度的影响，强调了集中速率和复杂度度量的重要性，并发现了混合速度对集中速率的影响，引入了相变的概念。

    

    本文针对具有任意混合率的一般混合随机过程提供了一个样本均值的上确界的界限。无论混合的速度如何，该界限由一个集中速率和一种新颖的复杂度度量组成。然而，混合的速度影响前者的数量，意味着出现了相变。快速混合导致标准的根号n集中速率，而慢速混合导致较慢的集中速率，其速度取决于混合结构。我们的发现应用于推导具有任意混合率的一般混合过程的强逼近结果。

    arXiv:2402.11394v1 Announce Type: cross  Abstract: This paper provides a bound for the supremum of sample averages over a class of functions for a general class of mixing stochastic processes with arbitrary mixing rates. Regardless of the speed of mixing, the bound is comprised of a concentration rate and a novel measure of complexity. The speed of mixing, however, affects the former quantity implying a phase transition. Fast mixing leads to the standard root-n concentration rate, while slow mixing leads to a slower concentration rate, its speed depends on the mixing structure. Our findings are applied to derive strong approximation results for a general class of mixing processes with arbitrary mixing rates.
    
[^2]: 随机赌博机中的遗憾分布：期望和尾部风险之间的最优权衡

    Regret Distribution in Stochastic Bandits: Optimal Trade-off between Expectation and Tail Risk. (arXiv:2304.04341v1 [stat.ML])

    [http://arxiv.org/abs/2304.04341](http://arxiv.org/abs/2304.04341)

    该论文探讨了随机多臂赌博问题中，如何在期望和尾部风险之间做出最优权衡。提出了一种新的策略，能够实现最坏和实例相关的优异表现，并且能够最小化遗憾尾部概率。

    

    本文研究了随机多臂赌博问题中，遗憾分布的期望和尾部风险之间的权衡问题。我们完全刻画了策略设计中三个期望性质之间的相互作用：最坏情况下的最优性，实例相关的一致性和轻尾风险。我们展示了期望遗憾的顺序如何影响遗憾尾部概率的衰减率，同时包括了最坏情况和实例相关的情况。我们提出了一种新的策略，以表征对于任何遗憾阈值的最优遗憾尾部概率。具体地，对于任何给定的$\alpha \in [1/2, 1)$和$\beta \in [0, \alpha]$，我们的策略可以实现平均期望遗憾$\tilde O(T^\alpha)$的最坏情况下$\alpha$-最优和期望遗憾$\tilde O(T^\beta)$的实例相关的$\beta$-一致性，并且享有一定的概率可以避免$\tilde O(T^\delta)$的遗憾($\delta \geq \alpha$在最坏情况下和$\delta \geq \beta$在实例相关的情况下)。

    We study the trade-off between expectation and tail risk for regret distribution in the stochastic multi-armed bandit problem. We fully characterize the interplay among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to characterize the optimal regret tail probability for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, \alpha]$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ (we call it $\alpha$-optimal) and an instance-dependent expected regret of $\tilde O(T^\beta)$ (we call it $\beta$-consistent), while enjoys a probability of incurring an $\tilde O(T^\delta)$ regret ($\delta\geq\alpha$ in the worst-case scenario and $\delta\geq\beta$ in the instance-dependent s
    

