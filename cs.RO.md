# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space](https://arxiv.org/abs/2404.01752) | 提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案 |
| [^2] | [Equivariant Ensembles and Regularization for Reinforcement Learning in Map-based Path Planning](https://arxiv.org/abs/2403.12856) | 本文提出了一种无需专门神经网络组件的等变策略和不变值函数构建方法，在基于地图的路径规划中展示了等变集合和正则化如何提高样本效率和性能 |
| [^3] | [Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control.](http://arxiv.org/abs/2401.16889) | 本论文使用深度强化学习创建了一种通用的双足机器人动态运动控制器，该控制器可以应用于多种动态双足技能，并且在模拟环境和实际环境中展现出了优越性能。 |

# 详细

[^1]: 安全间隔RRT*用于连续空间中可扩展的多机器人路径规划

    Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space

    [https://arxiv.org/abs/2404.01752](https://arxiv.org/abs/2404.01752)

    提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案

    

    在本文中，我们考虑了在连续空间中解决多机器人路径规划（MRPP）问题以找到无冲突路径的问题。问题的困难主要来自两个因素。首先，涉及多个机器人会导致组合决策，使搜索空间呈指数级增长。其次，连续空间呈现出潜在无限的状态和动作。针对这个问题，我们提出了一个两级方法，低级是基于采样的规划器安全间隔RRT*（SI-RRT*），用于找到单个机器人的无碰撞轨迹。高级可以使用能够解决机器人间冲突的任何方法，我们采用了两种代表性方法，即优先规划（SI-CPP）和基于冲突的搜索（SI-CCBS）。实验结果表明，SI-RRT*能够快速找到高质量解决方案，并且所需的样本数量较少。SI-CPP表现出更好的可扩展性，而SI-CCBS产生

    arXiv:2404.01752v1 Announce Type: cross  Abstract: In this paper, we consider the problem of Multi-Robot Path Planning (MRPP) in continuous space to find conflict-free paths. The difficulty of the problem arises from two primary factors. First, the involvement of multiple robots leads to combinatorial decision-making, which escalates the search space exponentially. Second, the continuous space presents potentially infinite states and actions. For this problem, we propose a two-level approach where the low level is a sampling-based planner Safe Interval RRT* (SI-RRT*) that finds a collision-free trajectory for individual robots. The high level can use any method that can resolve inter-robot conflicts where we employ two representative methods that are Prioritized Planning (SI-CPP) and Conflict Based Search (SI-CCBS). Experimental results show that SI-RRT* can find a high-quality solution quickly with a small number of samples. SI-CPP exhibits improved scalability while SI-CCBS produces 
    
[^2]: 基于地图的路径规划中的等变集合和正则化的强化学习

    Equivariant Ensembles and Regularization for Reinforcement Learning in Map-based Path Planning

    [https://arxiv.org/abs/2403.12856](https://arxiv.org/abs/2403.12856)

    本文提出了一种无需专门神经网络组件的等变策略和不变值函数构建方法，在基于地图的路径规划中展示了等变集合和正则化如何提高样本效率和性能

    

    在强化学习（RL）中，利用环境的对称性可以显著增强效率、鲁棒性和性能。然而，确保深度RL策略和值网络分别是等变和不变的以利用这些对称性是一个重大挑战。相关工作尝试通过构造具有等变性和不变性的网络来设计，这限制了它们只能使用非常受限的组件库，进而阻碍了网络的表现能力。本文提出了一种构建等变策略和不变值函数的方法，而无需专门的神经网络组件，我们将其称为等变集合。我们进一步添加了一个正则化项，用于在训练过程中增加归纳偏差。在基于地图的路径规划案例研究中，我们展示了等变集合和正则化如何有益于样本效率和性能。

    arXiv:2403.12856v1 Announce Type: new  Abstract: In reinforcement learning (RL), exploiting environmental symmetries can significantly enhance efficiency, robustness, and performance. However, ensuring that the deep RL policy and value networks are respectively equivariant and invariant to exploit these symmetries is a substantial challenge. Related works try to design networks that are equivariant and invariant by construction, limiting them to a very restricted library of components, which in turn hampers the expressiveness of the networks. This paper proposes a method to construct equivariant policies and invariant value functions without specialized neural network components, which we term equivariant ensembles. We further add a regularization term for adding inductive bias during training. In a map-based path planning case study, we show how equivariant ensembles and regularization benefit sample efficiency and performance.
    
[^3]: 强化学习用于多功能、动态和稳健的双足运动控制

    Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control. (arXiv:2401.16889v1 [cs.RO])

    [http://arxiv.org/abs/2401.16889](http://arxiv.org/abs/2401.16889)

    本论文使用深度强化学习创建了一种通用的双足机器人动态运动控制器，该控制器可以应用于多种动态双足技能，并且在模拟环境和实际环境中展现出了优越性能。

    

    本论文提出了一项关于使用深度强化学习（RL）创建双足机器人动态运动控制器的综合研究。我们不仅仅专注于单一的运动技能，而是开发了一种通用的控制解决方案，可以用于一系列动态双足技能，从周期性行走和奔跑到非周期性跳跃和站立。我们基于RL的控制器采用了一种新颖的双历史架构，利用机器人的长期和短期输入/输出（I/O）历史。通过提出的端到端RL方法进行训练时，这种控制架构在模拟环境和实际环境中的多样化技能上始终表现优于其他方法。该研究还深入探讨了所提出的RL系统在开发运动控制器方面引入的适应性和稳健性。我们证明了所提出的架构可以适应时间不变的动力学变化和时间变化的变化，如接触事件，通过有效地

    This paper presents a comprehensive study on using deep reinforcement learning (RL) to create dynamic locomotion controllers for bipedal robots. Going beyond focusing on a single locomotion skill, we develop a general control solution that can be used for a range of dynamic bipedal skills, from periodic walking and running to aperiodic jumping and standing. Our RL-based controller incorporates a novel dual-history architecture, utilizing both a long-term and short-term input/output (I/O) history of the robot. This control architecture, when trained through the proposed end-to-end RL approach, consistently outperforms other methods across a diverse range of skills in both simulation and the real world.The study also delves into the adaptivity and robustness introduced by the proposed RL system in developing locomotion controllers. We demonstrate that the proposed architecture can adapt to both time-invariant dynamics shifts and time-variant changes, such as contact events, by effectivel
    

