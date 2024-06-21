# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Control of Logically Constrained Partially Observable and Multi-Agent Markov Decision Processes.](http://arxiv.org/abs/2305.14736) | 本文介绍了一个用于部分可观察和多智能体马尔可夫决策过程的最优控制理论，能够使用时间逻辑规范表达约束，并提供了一种结构化的方法来合成策略以最大化累积奖励并保证约束条件的概率足够高。同时我们还提供了对信息不对称的多智能体设置进行最优控制的框架。 |

# 详细

[^1]: 逻辑约束下的部分可观察和多智能体马尔可夫决策过程的最优控制

    Optimal Control of Logically Constrained Partially Observable and Multi-Agent Markov Decision Processes. (arXiv:2305.14736v1 [cs.AI])

    [http://arxiv.org/abs/2305.14736](http://arxiv.org/abs/2305.14736)

    本文介绍了一个用于部分可观察和多智能体马尔可夫决策过程的最优控制理论，能够使用时间逻辑规范表达约束，并提供了一种结构化的方法来合成策略以最大化累积奖励并保证约束条件的概率足够高。同时我们还提供了对信息不对称的多智能体设置进行最优控制的框架。

    

    自动化系统通常会产生逻辑约束，例如来自安全、操作或法规要求，可以用时间逻辑规范表达这些约束。系统状态通常是部分可观察的，可能包含具有共同目标但不同信息结构和约束的多个智能体。在本文中，我们首先引入了一个最优控制理论，用于具有有限线性时间逻辑约束的部分可观察马尔可夫决策过程（POMDP）。我们提供了一种结构化方法，用于合成策略，同时确保满足时间逻辑约束的概率足够高时最大化累积回报。我们的方法具有关于近似奖励最优性和约束满足的保证。然后我们在此基础上构建了一个对信息不对称的具有逻辑约束的多智能体设置进行最优控制的框架。我们阐述了该方法并给出了理论保证。

    Autonomous systems often have logical constraints arising, for example, from safety, operational, or regulatory requirements. Such constraints can be expressed using temporal logic specifications. The system state is often partially observable. Moreover, it could encompass a team of multiple agents with a common objective but disparate information structures and constraints. In this paper, we first introduce an optimal control theory for partially observable Markov decision processes (POMDPs) with finite linear temporal logic constraints. We provide a structured methodology for synthesizing policies that maximize a cumulative reward while ensuring that the probability of satisfying a temporal logic constraint is sufficiently high. Our approach comes with guarantees on approximate reward optimality and constraint satisfaction. We then build on this approach to design an optimal control framework for logically constrained multi-agent settings with information asymmetry. We illustrate the
    

