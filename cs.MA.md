# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Congestion Pricing for Efficiency and Equity: Theory and Applications to the San Francisco Bay Area.](http://arxiv.org/abs/2401.16844) | 本研究提出了一种新的拥堵定价方案，旨在同时减少交通拥堵水平和缩小不同旅行者之间的成本差异，从而提高效率和公平性。 |
| [^2] | [Curriculum Learning for Relative Overgeneralization.](http://arxiv.org/abs/2212.02733) | 本论文提出了一种名为相对过度泛化的课程学习（CURO）的新算法来解决多智能体强化学习中存在的相对过度泛化 (RO) 问题，该方法在解决展示强RO的合作任务方面具有很好的表现。 |
| [^3] | [D3G: Learning Multi-robot Coordination from Demonstrations.](http://arxiv.org/abs/2207.08892) | 本文提出了一个D3G框架，可以从演示中学习多机器人协调。通过最小化轨迹与演示之间的不匹配，每个机器人可以自动调整其个体动态和目标，提高了学习效率和效果。 |

# 详细

[^1]: 用于效率和公平性的拥堵定价：理论及其在旧金山湾区的应用

    Congestion Pricing for Efficiency and Equity: Theory and Applications to the San Francisco Bay Area. (arXiv:2401.16844v1 [cs.GT])

    [http://arxiv.org/abs/2401.16844](http://arxiv.org/abs/2401.16844)

    本研究提出了一种新的拥堵定价方案，旨在同时减少交通拥堵水平和缩小不同旅行者之间的成本差异，从而提高效率和公平性。

    

    拥堵定价被许多城市用于缓解交通拥堵，但由于对低收入旅行者影响较大，引发了关于社会经济差距扩大的担忧。在本研究中，我们提出了一种新的拥堵定价方案，不仅可以最大限度地减少交通拥堵，还可以将公平性目标纳入其中，以减少不同支付意愿的旅行者之间的成本差异。我们的分析基于一个具有异质旅行者群体的拥堵博弈模型。我们提出了四种考虑实际因素的定价方案，例如对不同旅行者群体收取差异化的通行费以及征收整个路网中的所有边或只征收其中一部分边的选择。我们在旧金山湾区的校准高速公路网络中评估了我们的定价方案。我们证明了拥堵定价方案可以提高效率（即减少平均旅行时间）和公平性。

    Congestion pricing, while adopted by many cities to alleviate traffic congestion, raises concerns about widening socioeconomic disparities due to its disproportionate impact on low-income travelers. In this study, we address this concern by proposing a new class of congestion pricing schemes that not only minimize congestion levels but also incorporate an equity objective to reduce cost disparities among travelers with different willingness-to-pay. Our analysis builds on a congestion game model with heterogeneous traveler populations. We present four pricing schemes that account for practical considerations, such as the ability to charge differentiated tolls to various traveler populations and the option to toll all or only a subset of edges in the network. We evaluate our pricing schemes in the calibrated freeway network of the San Francisco Bay Area. We demonstrate that the proposed congestion pricing schemes improve both efficiency (in terms of reduced average travel time) and equit
    
[^2]: 相对过度泛化的课程学习

    Curriculum Learning for Relative Overgeneralization. (arXiv:2212.02733v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.02733](http://arxiv.org/abs/2212.02733)

    本论文提出了一种名为相对过度泛化的课程学习（CURO）的新算法来解决多智能体强化学习中存在的相对过度泛化 (RO) 问题，该方法在解决展示强RO的合作任务方面具有很好的表现。

    

    在多智能体强化学习 (MARL) 中，许多流行方法如 VDN 和 QMIX，都容易受到相对过度泛化 (RO) 这一关键性的多智能体病理的影响。当合作任务中最佳联合行动的效用低于次优联合行动时，就会出现RO。RO可能导致智能体陷入局部最优解或无法解决需要智能体之间在给定时间步长内进行大量协调的合作任务。最近的基于价值的MARL算法，如QPLEX和WQMIX可以在一定程度上克服RO。然而，我们的实验结果表明，它们仍然无法解决展示强RO的合作任务。在这项工作中，我们提出了一种称为相对过度泛化的课程学习（CURO）的新方法，以更好地克服RO。在CURO中，我们首先微调目标任务的奖励函数以生成适合当前能力的源任务来解决展示强RO的目标任务。

    In multi-agent reinforcement learning (MARL), many popular methods, such as VDN and QMIX, are susceptible to a critical multi-agent pathology known as relative overgeneralization (RO), which arises when the optimal joint action's utility falls below that of a sub-optimal joint action in cooperative tasks. RO can cause the agents to get stuck into local optima or fail to solve cooperative tasks that require significant coordination between agents within a given timestep. Recent value-based MARL algorithms such as QPLEX and WQMIX can overcome RO to some extent. However, our experimental results show that they can still fail to solve cooperative tasks that exhibit strong RO. In this work, we propose a novel approach called curriculum learning for relative overgeneralization (CURO) to better overcome RO. To solve a target task that exhibits strong RO, in CURO, we first fine-tune the reward function of the target task to generate source tasks that are tailored to the current ability of the 
    
[^3]: D3G: 从演示中学习多机器人协调

    D3G: Learning Multi-robot Coordination from Demonstrations. (arXiv:2207.08892v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2207.08892](http://arxiv.org/abs/2207.08892)

    本文提出了一个D3G框架，可以从演示中学习多机器人协调。通过最小化轨迹与演示之间的不匹配，每个机器人可以自动调整其个体动态和目标，提高了学习效率和效果。

    

    本文开发了一个分布式可微动态游戏（D3G）框架，可以实现从演示中学习多机器人协调。我们将多机器人协调表示为一个动态游戏，其中一个机器人的行为受其自身动态和目标的控制，同时也取决于其他机器人的行为。因此，通过调整每个机器人的目标和动态，可以适应协调。所提出的D3G使每个机器人通过最小化其轨迹与演示之间的不匹配，在分布式方式下自动调整其个体动态和目标。该学习框架具有新的设计，包括一个前向传递，所有机器人合作寻找游戏的纳什均衡，以及一个反向传递，在通信图中传播梯度。我们在仿真中测试了D3G，并给出了不同任务配置的两种机器人。结果证明了D3G学习多机器人协调的能力。

    This paper develops a Distributed Differentiable Dynamic Game (D3G) framework, which enables learning multi-robot coordination from demonstrations. We represent multi-robot coordination as a dynamic game, where the behavior of a robot is dictated by its own dynamics and objective that also depends on others' behavior. The coordination thus can be adapted by tuning the objective and dynamics of each robot. The proposed D3G enables each robot to automatically tune its individual dynamics and objectives in a distributed manner by minimizing the mismatch between its trajectory and demonstrations. This learning framework features a new design, including a forward-pass, where all robots collaboratively seek Nash equilibrium of a game, and a backward-pass, where gradients are propagated via the communication graph. We test the D3G in simulation with two types of robots given different task configurations. The results validate the capability of D3G for learning multi-robot coordination from de
    

