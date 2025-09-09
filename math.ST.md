# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Rate-Optimal Partitioning Classification from Observable and from Privatised Data](https://arxiv.org/abs/2312.14889) | 研究了在放宽条件下的分区分类方法的收敛速率，提出了绝对连续分量的新特性，计算了分类错误概率的精确收敛率 |
| [^2] | [The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model.](http://arxiv.org/abs/2305.16589) | 本文研究了强化学习中的模型鲁棒性以缩小模拟与真实差距，提出了一个名为“分布鲁棒值迭代”的基于模型的方法，可以优化最坏情况下的表现。 |

# 详细

[^1]: 论从可观测和私密数据中实现速率最优分区分类

    On Rate-Optimal Partitioning Classification from Observable and from Privatised Data

    [https://arxiv.org/abs/2312.14889](https://arxiv.org/abs/2312.14889)

    研究了在放宽条件下的分区分类方法的收敛速率，提出了绝对连续分量的新特性，计算了分类错误概率的精确收敛率

    

    在这篇论文中，我们重新审视了分区分类的经典方法，并研究了在放宽条件下的收敛速率，包括可观测（非私密）和私密数据。我们假设特征向量$X$取值于$\mathbb{R}^d$，其标签为$Y$。之前关于分区分类器的结果基于强密度假设，这种假设限制较大，我们通过简单的例子加以证明。我们假设$X$的分布是绝对连续分布和离散分布的混合体，其中绝对连续分量集中于一个$d_a$维子空间。在这里，我们在更宽松的条件下研究了这个问题：除了标准的Lipschitz和边际条件外，我们还引入了绝对连续分量的一个新特性，通过该特性计算了分类错误概率的精确收敛率，对于...

    arXiv:2312.14889v2 Announce Type: replace-cross  Abstract: In this paper we revisit the classical method of partitioning classification and study its convergence rate under relaxed conditions, both for observable (non-privatised) and for privatised data. Let the feature vector $X$ take values in $\mathbb{R}^d$ and denote its label by $Y$. Previous results on the partitioning classifier worked with the strong density assumption, which is restrictive, as we demonstrate through simple examples. We assume that the distribution of $X$ is a mixture of an absolutely continuous and a discrete distribution, such that the absolutely continuous component is concentrated to a $d_a$ dimensional subspace. Here, we study the problem under much milder assumptions: in addition to the standard Lipschitz and margin conditions, a novel characteristic of the absolutely continuous component is introduced, by which the exact convergence rate of the classification error probability is calculated, both for the
    
[^2]: 具有生成模型的强化学习中分布鲁棒性的可疑价格

    The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model. (arXiv:2305.16589v1 [cs.LG])

    [http://arxiv.org/abs/2305.16589](http://arxiv.org/abs/2305.16589)

    本文研究了强化学习中的模型鲁棒性以缩小模拟与真实差距，提出了一个名为“分布鲁棒值迭代”的基于模型的方法，可以优化最坏情况下的表现。

    

    本文研究了强化学习中的模型鲁棒性，以减少在实践中的模拟与真实差距。我们采用分布鲁棒马尔可夫决策过程（RMDPs）框架，旨在学习一个策略，在部署环境落在预定的不确定性集合内时，优化最坏情况下的表现。尽管最近有了一些努力，但RMDPs的样本复杂度仍然没有得到解决，无论使用的不确定性集合是什么。不清楚分布鲁棒性与标准强化学习相比是否具有统计学上的影响。假设有一个生成模型，根据名义MDP绘制样本，我们将描述RMDPs的样本复杂度，当由总变差（TV）距离或$\chi^2$分歧指定不确定性集合时。在这里研究的算法是一种基于模型的方法，称为分布鲁棒值迭代，证明了它在整个范围内都是近乎最优的。

    This paper investigates model robustness in reinforcement learning (RL) to reduce the sim-to-real gap in practice. We adopt the framework of distributionally robust Markov decision processes (RMDPs), aimed at learning a policy that optimizes the worst-case performance when the deployed environment falls within a prescribed uncertainty set around the nominal MDP. Despite recent efforts, the sample complexity of RMDPs remained mostly unsettled regardless of the uncertainty set in use. It was unclear if distributional robustness bears any statistical consequences when benchmarked against standard RL.  Assuming access to a generative model that draws samples based on the nominal MDP, we characterize the sample complexity of RMDPs when the uncertainty set is specified via either the total variation (TV) distance or $\chi^2$ divergence. The algorithm studied here is a model-based method called {\em distributionally robust value iteration}, which is shown to be near-optimal for the full range
    

