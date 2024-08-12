# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Peer Adaptation with Context-aware Exploration](https://arxiv.org/abs/2402.02468) | 本文提出了一种基于上下文感知的探索方法，用于快速适应具有不同策略的未知同伴。通过奖励智能体在历史上下文中有效识别同伴行为模式，该方法能够促进智能体积极探索和快速适应，从而在不确定同伴策略时收集信息反馈，并在有信心时利用上下文执行最佳反应。 |
| [^2] | [Crowd-Aware Multi-Agent Pathfinding With Boosted Curriculum Reinforcement Learning.](http://arxiv.org/abs/2309.10275) | 该论文介绍了CRAMP，一种基于增强式课程强化学习的众包感知分散式路径规划方法，旨在解决拥挤环境下多智能体路径规划的困难。 |

# 详细

[^1]: 基于上下文感知探索的快速适应未知同伴

    Fast Peer Adaptation with Context-aware Exploration

    [https://arxiv.org/abs/2402.02468](https://arxiv.org/abs/2402.02468)

    本文提出了一种基于上下文感知的探索方法，用于快速适应具有不同策略的未知同伴。通过奖励智能体在历史上下文中有效识别同伴行为模式，该方法能够促进智能体积极探索和快速适应，从而在不确定同伴策略时收集信息反馈，并在有信心时利用上下文执行最佳反应。

    

    在多智能体游戏中，快速适应具有不同策略的未知同伴是一个关键挑战。为了做到这一点，智能体能够高效地探索和识别同伴的策略至关重要，因为这是适应中进行最佳反应的先决条件。然而，当游戏是部分可观测且时间跨度很长时，探索未知同伴的策略是困难的。在本文中，我们提出了一种同伴识别奖励，根据智能体在历史环境下（例如多个回合的观察）如何有效地识别同伴的行为模式来奖励学习智能体。这个奖励激励智能体学习一种基于上下文的策略，以实现有效探索和快速适应，即在对同伴策略不确定时积极寻找和收集信息反馈，并在有信心时利用上下文执行最佳反应。我们在不同的测试场景上进行了评估。

    Fast adapting to unknown peers (partners or opponents) with different strategies is a key challenge in multi-agent games. To do so, it is crucial for the agent to efficiently probe and identify the peer's strategy, as this is the prerequisite for carrying out the best response in adaptation. However, it is difficult to explore the strategies of unknown peers, especially when the games are partially observable and have a long horizon. In this paper, we propose a peer identification reward, which rewards the learning agent based on how well it can identify the behavior pattern of the peer over the historical context, such as the observation over multiple episodes. This reward motivates the agent to learn a context-aware policy for effective exploration and fast adaptation, i.e., to actively seek and collect informative feedback from peers when uncertain about their policies and to exploit the context to perform the best response when confident. We evaluate our method on diverse testbeds 
    
[^2]: 具有增强课程强化学习的众包感知多智能体路径规划研究

    Crowd-Aware Multi-Agent Pathfinding With Boosted Curriculum Reinforcement Learning. (arXiv:2309.10275v1 [cs.RO])

    [http://arxiv.org/abs/2309.10275](http://arxiv.org/abs/2309.10275)

    该论文介绍了CRAMP，一种基于增强式课程强化学习的众包感知分散式路径规划方法，旨在解决拥挤环境下多智能体路径规划的困难。

    

    在拥挤环境中进行的多智能体路径规划是一个具有挑战性的运动规划问题，旨在为系统中的所有智能体找到无碰撞路径。多智能体路径规划在各个领域中都有广泛的应用，包括空中群体、自动化仓储机器人和自动驾驶车辆。当前的多智能体路径规划方法可以大致分为两种主要类别：集中式规划和分散式规划。集中式规划受到维度灾难的困扰，因此在大型和复杂环境中不具备良好的可扩展性。另一方面，分散式规划使智能体能够在部分可观察环境中进行实时路径规划，展示了隐式的协调能力。然而，在密集环境中它们的收敛速度较慢且性能下降。本文介绍了一种名为CRAMP的众包感知分散式方法，通过增强式课程引导的强化学习来解决这个问题。

    Multi-Agent Path Finding (MAPF) in crowded environments presents a challenging problem in motion planning, aiming to find collision-free paths for all agents in the system. MAPF finds a wide range of applications in various domains, including aerial swarms, autonomous warehouse robotics, and self-driving vehicles. The current approaches for MAPF can be broadly categorized into two main categories: centralized and decentralized planning. Centralized planning suffers from the curse of dimensionality and thus does not scale well in large and complex environments. On the other hand, decentralized planning enables agents to engage in real-time path planning within a partially observable environment, demonstrating implicit coordination. However, they suffer from slow convergence and performance degradation in dense environments. In this paper, we introduce CRAMP, a crowd-aware decentralized approach to address this problem by leveraging reinforcement learning guided by a boosted curriculum-b
    

