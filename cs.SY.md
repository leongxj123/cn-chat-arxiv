# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Regret Analysis of the Posterior Sampling-based Learning Algorithm for Episodic POMDPs.](http://arxiv.org/abs/2310.10107) | 本文分析了后验采样学习算法在序列化POMDPs中的遗憾性能，并在一定条件下提供了改进的多项式贝叶斯遗憾界。 |

# 详细

[^1]: 后验采样学习算法在序列化POMDPs中的遗憾分析

    Regret Analysis of the Posterior Sampling-based Learning Algorithm for Episodic POMDPs. (arXiv:2310.10107v1 [cs.LG])

    [http://arxiv.org/abs/2310.10107](http://arxiv.org/abs/2310.10107)

    本文分析了后验采样学习算法在序列化POMDPs中的遗憾性能，并在一定条件下提供了改进的多项式贝叶斯遗憾界。

    

    相比于马尔科夫决策过程（MDPs），部分可观察马尔科夫决策过程（POMDPs）的学习由于观察数据难以解读而变得更加困难。在本文中，我们考虑了具有未知转移和观测模型的POMDPs中的序列化学习问题。我们考虑了基于后验采样的强化学习算法（PSRL）在POMDPs中的应用，并证明其贝叶斯遗憾随着序列的数量的平方根而缩小。一般来说，遗憾随着时间长度$H$呈指数级增长，并通过提供一个下界证明了这一点。然而，在POMDP是欠完备且弱可识别的条件下，我们建立了一个多项式贝叶斯遗憾界，相比于arXiv:2204.08967的最新结果，改进了遗憾界约$\Omega(H^2\sqrt{SA})$倍。

    Compared to Markov Decision Processes (MDPs), learning in Partially Observable Markov Decision Processes (POMDPs) can be significantly harder due to the difficulty of interpreting observations. In this paper, we consider episodic learning problems in POMDPs with unknown transition and observation models. We consider the Posterior Sampling-based Reinforcement Learning (PSRL) algorithm for POMDPs and show that its Bayesian regret scales as the square root of the number of episodes. In general, the regret scales exponentially with the horizon length $H$, and we show that this is inevitable by providing a lower bound. However, under the condition that the POMDP is undercomplete and weakly revealing, we establish a polynomial Bayesian regret bound that improves the regret bound by a factor of $\Omega(H^2\sqrt{SA})$ over the recent result by arXiv:2204.08967.
    

