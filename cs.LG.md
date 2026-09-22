# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tsallis Entropy Regularization for Linearly Solvable MDP and Linear Quadratic Regulator](https://arxiv.org/abs/2403.01805) | 本文介绍了Tsallis熵正则化方法，用于解决线性可解MDP和线性二次调节器问题，能够平衡探索和获得的控制法则的稀疏性。 |
| [^2] | [Subgoal Search For Complex Reasoning Tasks](https://arxiv.org/abs/2108.11204) | 提出了子目标搜索（kSubS）方法，通过学习的子目标生成器产生多样性的子目标，减少搜索空间并在Sokoban、魔方和不等式证明三个领域取得了强大的结果。 |
| [^3] | [On the Effects of Data Heterogeneity on the Convergence Rates of Distributed Linear System Solvers.](http://arxiv.org/abs/2304.10640) | 本文比较了投影方法和优化方法求解分布式线性系统的收敛速度，提出了角异构性的几何概念，并对最有效的算法(APC和D-HBM)的收敛速度进行了约束和比较。 |
| [^4] | [Optimal Sample Complexity of Reinforcement Learning for Mixing Discounted Markov Decision Processes.](http://arxiv.org/abs/2302.07477) | 这篇论文研究了对于混合折扣马尔可夫决策过程的强化学习的最优样本复杂度理论。作者发现，在混合的情况下，最优样本复杂度依赖于总变异混合时间、折扣因子和解误差容忍度。 |
| [^5] | [Combinatorial Inference on the Optimal Assortment in Multinomial Logit Models.](http://arxiv.org/abs/2301.12254) | 本文提出了一种基于多项式logit模型的推断框架，可以测试最优产品组合是否具有特定性质。 |

# 详细

[^1]: Tsallis熵正则化用于线性可解MDP和线性二次调节器

    Tsallis Entropy Regularization for Linearly Solvable MDP and Linear Quadratic Regulator

    [https://arxiv.org/abs/2403.01805](https://arxiv.org/abs/2403.01805)

    本文介绍了Tsallis熵正则化方法，用于解决线性可解MDP和线性二次调节器问题，能够平衡探索和获得的控制法则的稀疏性。

    

    Shannon熵正则化被广泛应用于最优控制中，因为它能促进探索并增强鲁棒性，例如，Soft Actor-Critic中采用的最大熵强化学习。本文使用Tsallis熵（Shannon熵的单参数扩展）来正则化线性可解MDP和线性二次调节器。我们推导出这些问题的解，并展示了它在平衡探索和获得的控制法则的稀疏性方面的实用性。

    arXiv:2403.01805v1 Announce Type: cross  Abstract: Shannon entropy regularization is widely adopted in optimal control due to its ability to promote exploration and enhance robustness, e.g., maximum entropy reinforcement learning known as Soft Actor-Critic. In this paper, Tsallis entropy, which is a one-parameter extension of Shannon entropy, is used for the regularization of linearly solvable MDP and linear quadratic regulators. We derive the solution for these problems and demonstrate its usefulness in balancing between exploration and sparsity of the obtained control law.
    
[^2]: 复杂推理任务的子目标搜索

    Subgoal Search For Complex Reasoning Tasks

    [https://arxiv.org/abs/2108.11204](https://arxiv.org/abs/2108.11204)

    提出了子目标搜索（kSubS）方法，通过学习的子目标生成器产生多样性的子目标，减少搜索空间并在Sokoban、魔方和不等式证明三个领域取得了强大的结果。

    

    人类擅长通过从一个想法移动到相关的想法的思维过程来解决复杂的推理任务。受此启发，我们提出了子目标搜索（kSubS）方法。其关键组件是一个学习的子目标生成器，产生多样性的既可实现又接近解决方案的子目标。使用子目标可以减少搜索空间，并引入适合高效规划的高级搜索图。本文中，我们使用基于Transformer的子目标模块结合经典的最佳优先搜索框架来实现kSubS。我们展示了一种简单的生成第$k$步子目标的方法在三个具有挑战性的领域上表现出惊人的效率：两个流行的益智游戏Sokoban和魔方以及不等式证明基准INT。kSubS在适度的计算预算内取得了强大的结果，包括在INT上的最新成果。

    arXiv:2108.11204v3 Announce Type: replace  Abstract: Humans excel in solving complex reasoning tasks through a mental process of moving from one idea to a related one. Inspired by this, we propose Subgoal Search (kSubS) method. Its key component is a learned subgoal generator that produces a diversity of subgoals that are both achievable and closer to the solution. Using subgoals reduces the search space and induces a high-level search graph suitable for efficient planning. In this paper, we implement kSubS using a transformer-based subgoal module coupled with the classical best-first search framework. We show that a simple approach of generating $k$-th step ahead subgoals is surprisingly efficient on three challenging domains: two popular puzzle games, Sokoban and the Rubik's Cube, and an inequality proving benchmark INT. kSubS achieves strong results including state-of-the-art on INT within a modest computational budget.
    
[^3]: 论数据异构性对分布式线性系统求解器收敛速度的影响

    On the Effects of Data Heterogeneity on the Convergence Rates of Distributed Linear System Solvers. (arXiv:2304.10640v1 [cs.DC])

    [http://arxiv.org/abs/2304.10640](http://arxiv.org/abs/2304.10640)

    本文比较了投影方法和优化方法求解分布式线性系统的收敛速度，提出了角异构性的几何概念，并对最有效的算法(APC和D-HBM)的收敛速度进行了约束和比较。

    

    本文考虑了解决大规模线性方程组的基本问题。特别地，我们考虑任务负责人打算在一组具有一些方程组子集的机器的分布式/联合帮助下解决该系统的设置。虽然有几种方法用于解决这个问题，但缺少对投影方法和优化方法收敛速度的严格比较。在本文中，我们分析并比较这两类算法，特别关注每个类别中最有效的方法，即最近提出的加速投影一致性(APC)和分布式重球方法(D-HBM)。为此，我们首先提出了称为角异构性的几何概念，并讨论其普遍性。使用该概念，我们约束并比较所研究算法的收敛速度，并捕捉两种方法的异构数据的效应。

    We consider the fundamental problem of solving a large-scale system of linear equations. In particular, we consider the setting where a taskmaster intends to solve the system in a distributed/federated fashion with the help of a set of machines, who each have a subset of the equations. Although there exist several approaches for solving this problem, missing is a rigorous comparison between the convergence rates of the projection-based methods and those of the optimization-based ones. In this paper, we analyze and compare these two classes of algorithms with a particular focus on the most efficient method from each class, namely, the recently proposed Accelerated Projection-Based Consensus (APC) and the Distributed Heavy-Ball Method (D-HBM). To this end, we first propose a geometric notion of data heterogeneity called angular heterogeneity and discuss its generality. Using this notion, we bound and compare the convergence rates of the studied algorithms and capture the effects of both 
    
[^4]: 对于混合折扣马尔可夫决策过程的强化学习的最优样本复杂度研究

    Optimal Sample Complexity of Reinforcement Learning for Mixing Discounted Markov Decision Processes. (arXiv:2302.07477v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.07477](http://arxiv.org/abs/2302.07477)

    这篇论文研究了对于混合折扣马尔可夫决策过程的强化学习的最优样本复杂度理论。作者发现，在混合的情况下，最优样本复杂度依赖于总变异混合时间、折扣因子和解误差容忍度。

    

    我们考虑了表格型强化学习（RL）对于在马尔可夫决策过程（MDP）中最大化无穷时间折扣奖励的最优样本复杂度理论。在这种设定下，已经为表格型问题开发了最优最坏情况复杂度结果，导致样本复杂度依赖于折扣系数$\gamma$和解误差容忍度$\epsilon$的形式为$\tilde \Theta((1-\gamma)^{-3}\epsilon^{-2})$，其中$\gamma$表示折扣因子，$\epsilon$为解误差容忍度。然而，在许多感兴趣的应用中，最优策略（或所有策略）会产生混合。我们确定，在这种情况下，最优样本复杂度的依赖关系为$\tilde \Theta(t_{\text{mix}}(1-\gamma)^{-2}\epsilon^{-2})$，其中$t_{\text{mix}}$是总变异混合时间。我们的分析基于再生型思想，我们认为这些思想对于研究一般状态空间MDPs的RL问题具有独立的兴趣。

    We consider the optimal sample complexity theory of tabular reinforcement learning (RL) for maximizing the infinite horizon discounted reward in a Markov decision process (MDP). Optimal worst-case complexity results have been developed for tabular RL problems in this setting, leading to a sample complexity dependence on $\gamma$ and $\epsilon$ of the form $\tilde \Theta((1-\gamma)^{-3}\epsilon^{-2})$, where $\gamma$ denotes the discount factor and $\epsilon$ is the solution error tolerance. However, in many applications of interest, the optimal policy (or all policies) induces mixing. We establish that in such settings, the optimal sample complexity dependence is $\tilde \Theta(t_{\text{mix}}(1-\gamma)^{-2}\epsilon^{-2})$, where $t_{\text{mix}}$ is the total variation mixing time. Our analysis is grounded in regeneration-type ideas, which we believe are of independent interest, as they can be used to study RL problems for general state space MDPs.
    
[^5]: 多项式Logit模型中最优产品组合的组合推断

    Combinatorial Inference on the Optimal Assortment in Multinomial Logit Models. (arXiv:2301.12254v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.12254](http://arxiv.org/abs/2301.12254)

    本文提出了一种基于多项式logit模型的推断框架，可以测试最优产品组合是否具有特定性质。

    

    最优的产品组合优化已经成为实践中的重要问题。本文提出了一种新的推断框架，用于测试最优产品组合是否具有特定性质。我们考虑了广泛采用的多项式logit（MNL）模型，并将其用于研究最优组合问题。

    Assortment optimization has received active explorations in the past few decades due to its practical importance. Despite the extensive literature dealing with optimization algorithms and latent score estimation, uncertainty quantification for the optimal assortment still needs to be explored and is of great practical significance. Instead of estimating and recovering the complete optimal offer set, decision-makers may only be interested in testing whether a given property holds true for the optimal assortment, such as whether they should include several products of interest in the optimal set, or how many categories of products the optimal set should include. This paper proposes a novel inferential framework for testing such properties. We consider the widely adopted multinomial logit (MNL) model, where we assume that each customer will purchase an item within the offered products with a probability proportional to the underlying preference score associated with the product. We reduce
    

