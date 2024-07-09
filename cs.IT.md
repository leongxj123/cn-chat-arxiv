# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Agent Deep Reinforcement Learning for Distributed Satellite Routing](https://arxiv.org/abs/2402.17666) | 本文介绍了一种多智能体深度强化学习方法，用于低地球轨道卫星星座中的路由，通过离线学习最佳路径，并在在线阶段进行高效的分布式路由。 |

# 详细

[^1]: 多智能体深度强化学习用于分布式卫星路由

    Multi-Agent Deep Reinforcement Learning for Distributed Satellite Routing

    [https://arxiv.org/abs/2402.17666](https://arxiv.org/abs/2402.17666)

    本文介绍了一种多智能体深度强化学习方法，用于低地球轨道卫星星座中的路由，通过离线学习最佳路径，并在在线阶段进行高效的分布式路由。

    

    本文介绍了一种用于低地球轨道卫星星座（LSatCs）中路由的多智能体深度强化学习（MA-DRL）方法。每个卫星是一个独立的决策制定智能体，具有对环境的部分知识，并受到附近智能体的反馈支持。在我们之前介绍的Q-routing解决方案的基础上，本文的贡献是将其扩展为一个深度学习框架，能够快速适应网络和交通变化，并基于两个阶段：（1）一个依赖全局深度神经网络（DNN）学习在每个可能位置和拥堵级别上的最佳路径的离线探索学习阶段；（2）一个带有本地、机载、预训练DNN的在线开发阶段。结果表明，MA-DRL能够有效地在离线学习最佳路由，然后加载以进行高效的分布式在线路由。

    arXiv:2402.17666v1 Announce Type: new  Abstract: This paper introduces a Multi-Agent Deep Reinforcement Learning (MA-DRL) approach for routing in Low Earth Orbit Satellite Constellations (LSatCs). Each satellite is an independent decision-making agent with a partial knowledge of the environment, and supported by feedback received from the nearby agents. Building on our previous work that introduced a Q-routing solution, the contribution of this paper is to extend it to a deep learning framework able to quickly adapt to the network and traffic changes, and based on two phases: (1) An offline exploration learning phase that relies on a global Deep Neural Network (DNN) to learn the optimal paths at each possible position and congestion level; (2) An online exploitation phase with local, on-board, pre-trained DNNs. Results show that MA-DRL efficiently learns optimal routes offline that are then loaded for an efficient distributed routing online.
    

