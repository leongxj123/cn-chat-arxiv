# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Sample Complexity of Reinforcement Learning for Mixing Discounted Markov Decision Processes.](http://arxiv.org/abs/2302.07477) | 这篇论文研究了对于混合折扣马尔可夫决策过程的强化学习的最优样本复杂度理论。作者发现，在混合的情况下，最优样本复杂度依赖于总变异混合时间、折扣因子和解误差容忍度。 |
| [^2] | [Combinatorial Inference on the Optimal Assortment in Multinomial Logit Models.](http://arxiv.org/abs/2301.12254) | 本文提出了一种基于多项式logit模型的推断框架，可以测试最优产品组合是否具有特定性质。 |

# 详细

[^1]: 对于混合折扣马尔可夫决策过程的强化学习的最优样本复杂度研究

    Optimal Sample Complexity of Reinforcement Learning for Mixing Discounted Markov Decision Processes. (arXiv:2302.07477v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.07477](http://arxiv.org/abs/2302.07477)

    这篇论文研究了对于混合折扣马尔可夫决策过程的强化学习的最优样本复杂度理论。作者发现，在混合的情况下，最优样本复杂度依赖于总变异混合时间、折扣因子和解误差容忍度。

    

    我们考虑了表格型强化学习（RL）对于在马尔可夫决策过程（MDP）中最大化无穷时间折扣奖励的最优样本复杂度理论。在这种设定下，已经为表格型问题开发了最优最坏情况复杂度结果，导致样本复杂度依赖于折扣系数$\gamma$和解误差容忍度$\epsilon$的形式为$\tilde \Theta((1-\gamma)^{-3}\epsilon^{-2})$，其中$\gamma$表示折扣因子，$\epsilon$为解误差容忍度。然而，在许多感兴趣的应用中，最优策略（或所有策略）会产生混合。我们确定，在这种情况下，最优样本复杂度的依赖关系为$\tilde \Theta(t_{\text{mix}}(1-\gamma)^{-2}\epsilon^{-2})$，其中$t_{\text{mix}}$是总变异混合时间。我们的分析基于再生型思想，我们认为这些思想对于研究一般状态空间MDPs的RL问题具有独立的兴趣。

    We consider the optimal sample complexity theory of tabular reinforcement learning (RL) for maximizing the infinite horizon discounted reward in a Markov decision process (MDP). Optimal worst-case complexity results have been developed for tabular RL problems in this setting, leading to a sample complexity dependence on $\gamma$ and $\epsilon$ of the form $\tilde \Theta((1-\gamma)^{-3}\epsilon^{-2})$, where $\gamma$ denotes the discount factor and $\epsilon$ is the solution error tolerance. However, in many applications of interest, the optimal policy (or all policies) induces mixing. We establish that in such settings, the optimal sample complexity dependence is $\tilde \Theta(t_{\text{mix}}(1-\gamma)^{-2}\epsilon^{-2})$, where $t_{\text{mix}}$ is the total variation mixing time. Our analysis is grounded in regeneration-type ideas, which we believe are of independent interest, as they can be used to study RL problems for general state space MDPs.
    
[^2]: 多项式Logit模型中最优产品组合的组合推断

    Combinatorial Inference on the Optimal Assortment in Multinomial Logit Models. (arXiv:2301.12254v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.12254](http://arxiv.org/abs/2301.12254)

    本文提出了一种基于多项式logit模型的推断框架，可以测试最优产品组合是否具有特定性质。

    

    最优的产品组合优化已经成为实践中的重要问题。本文提出了一种新的推断框架，用于测试最优产品组合是否具有特定性质。我们考虑了广泛采用的多项式logit（MNL）模型，并将其用于研究最优组合问题。

    Assortment optimization has received active explorations in the past few decades due to its practical importance. Despite the extensive literature dealing with optimization algorithms and latent score estimation, uncertainty quantification for the optimal assortment still needs to be explored and is of great practical significance. Instead of estimating and recovering the complete optimal offer set, decision-makers may only be interested in testing whether a given property holds true for the optimal assortment, such as whether they should include several products of interest in the optimal set, or how many categories of products the optimal set should include. This paper proposes a novel inferential framework for testing such properties. We consider the widely adopted multinomial logit (MNL) model, where we assume that each customer will purchase an item within the offered products with a probability proportional to the underlying preference score associated with the product. We reduce
    

