# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-agent reinforcement learning using echo-state network and its application to pedestrian dynamics](https://arxiv.org/abs/2312.11834) | 通过使用回声状态网络将行人实现为MARL代理，研究了他们学习避让其他代理向前移动的能力，在密度不太高的情况下取得成功。 |
| [^2] | [An active learning method for solving competitive multi-agent decision-making and control problems.](http://arxiv.org/abs/2212.12561) | 我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。 |

# 详细

[^1]: 使用回声状态网络的多智能体强化学习及其在行人动态中的应用

    Multi-agent reinforcement learning using echo-state network and its application to pedestrian dynamics

    [https://arxiv.org/abs/2312.11834](https://arxiv.org/abs/2312.11834)

    通过使用回声状态网络将行人实现为MARL代理，研究了他们学习避让其他代理向前移动的能力，在密度不太高的情况下取得成功。

    

    近年来，研究使用多智能体强化学习（MARL）模拟行人。本研究考虑了网格世界环境中的道路，并将行人实现为使用回声状态网络和最小二乘策略迭代方法的MARL代理。在这个环境下，研究了这些代理学习避开其他代理向前移动的能力。具体而言，我们考虑了两种任务：窄直接路径和宽绕道之间的选择，以及走廊中的双向行人流。模拟结果表明，当代理密度不太高时，学习是成功的。

    arXiv:2312.11834v2 Announce Type: replace-cross  Abstract: In recent years, simulations of pedestrians using the multi-agent reinforcement learning (MARL) have been studied. This study considered the roads on a grid-world environment, and implemented pedestrians as MARL agents using an echo-state network and the least squares policy iteration method. Under this environment, the ability of these agents to learn to move forward by avoiding other agents was investigated. Specifically, we considered two types of tasks: the choice between a narrow direct route and a broad detour, and the bidirectional pedestrian flow in a corridor. The simulations results indicated that the learning was successful when the density of the agents was not that high.
    
[^2]: 解决竞争性多智能体决策和控制问题的主动学习方法

    An active learning method for solving competitive multi-agent decision-making and control problems. (arXiv:2212.12561v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2212.12561](http://arxiv.org/abs/2212.12561)

    我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。

    

    我们提出了一种基于主动学习的方案，用于重构由相互作用代理人群体执行的私有策略，并预测底层多智能体交互过程的确切结果，这里被认为是一个稳定的行动配置文件。我们设想了一个场景，在这个场景中，一个具有学习程序的外部观察者可以通过私有的行动-反应映射进行查询和观察代理人的反应，集体的不动点对应于一个稳态配置文件。通过迭代地收集有意义的数据和更新行动-反应映射的参数估计，我们建立了评估所提出的主动学习方法的渐近性质的充分条件，以便如果收敛发生，它只能朝向一个稳态行动配置文件。这一事实导致了两个主要结果：i）学习局部精确的行动-反应映射替代物使得外部观察者能够成功完成其预测任务，ii）与代理人的互动提供了一种方法来优化策略以达到最佳效果。

    We propose a scheme based on active learning to reconstruct private strategies executed by a population of interacting agents and predict an exact outcome of the underlying multi-agent interaction process, here identified as a stationary action profile. We envision a scenario where an external observer, endowed with a learning procedure, can make queries and observe the agents' reactions through private action-reaction mappings, whose collective fixed point corresponds to a stationary profile. By iteratively collecting sensible data and updating parametric estimates of the action-reaction mappings, we establish sufficient conditions to assess the asymptotic properties of the proposed active learning methodology so that, if convergence happens, it can only be towards a stationary action profile. This fact yields two main consequences: i) learning locally-exact surrogates of the action-reaction mappings allows the external observer to succeed in its prediction task, and ii) working with 
    

