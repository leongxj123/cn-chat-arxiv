# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Reward: Learning Rewards via Conditional Video Diffusion](https://arxiv.org/abs/2312.14134) | 通过条件视频扩散学习奖励，解决复杂视觉强化学习问题，有效提高了任务成功率，超越了基线方法。 |
| [^2] | [Crowd-Aware Multi-Agent Pathfinding With Boosted Curriculum Reinforcement Learning.](http://arxiv.org/abs/2309.10275) | 该论文介绍了CRAMP，一种基于增强式课程强化学习的众包感知分散式路径规划方法，旨在解决拥挤环境下多智能体路径规划的困难。 |

# 详细

[^1]: 通过条件视频扩散学习奖励

    Diffusion Reward: Learning Rewards via Conditional Video Diffusion

    [https://arxiv.org/abs/2312.14134](https://arxiv.org/abs/2312.14134)

    通过条件视频扩散学习奖励，解决复杂视觉强化学习问题，有效提高了任务成功率，超越了基线方法。

    

    从专家视频中学习奖励为强化学习任务指定预期行为提供了一种经济高效的解决方案。本文提出了Diffusion Reward，这是一个通过条件视频扩散模型从专家视频中学习奖励以解决复杂视觉强化学习问题的新框架。我们的关键见解是，在专家轨迹的条件下观察到较低的生成多样性。因此，Diffusion Reward被形式化为负的条件熵，鼓励专家式行为的有效探索。我们展示了我们的方法在MetaWorld和Adroit的10个机器人操纵任务中以视觉输入和稀疏奖励的有效性。此外，Diffusion Reward甚至能够成功有效地解决未见过的任务，大大超越了基线方法。项目页面和代码：https://diffusion-reward.github.io/。

    arXiv:2312.14134v2 Announce Type: replace  Abstract: Learning rewards from expert videos offers an affordable and effective solution to specify the intended behaviors for reinforcement learning tasks. In this work, we propose Diffusion Reward, a novel framework that learns rewards from expert videos via conditional video diffusion models for solving complex visual RL problems. Our key insight is that lower generative diversity is observed when conditioned on expert trajectories. Diffusion Reward is accordingly formalized by the negative of conditional entropy that encourages productive exploration of expert-like behaviors. We show the efficacy of our method over 10 robotic manipulation tasks from MetaWorld and Adroit with visual input and sparse reward. Moreover, Diffusion Reward could even solve unseen tasks successfully and effectively, largely surpassing baseline methods. Project page and code: https://diffusion-reward.github.io/.
    
[^2]: 具有增强课程强化学习的众包感知多智能体路径规划研究

    Crowd-Aware Multi-Agent Pathfinding With Boosted Curriculum Reinforcement Learning. (arXiv:2309.10275v1 [cs.RO])

    [http://arxiv.org/abs/2309.10275](http://arxiv.org/abs/2309.10275)

    该论文介绍了CRAMP，一种基于增强式课程强化学习的众包感知分散式路径规划方法，旨在解决拥挤环境下多智能体路径规划的困难。

    

    在拥挤环境中进行的多智能体路径规划是一个具有挑战性的运动规划问题，旨在为系统中的所有智能体找到无碰撞路径。多智能体路径规划在各个领域中都有广泛的应用，包括空中群体、自动化仓储机器人和自动驾驶车辆。当前的多智能体路径规划方法可以大致分为两种主要类别：集中式规划和分散式规划。集中式规划受到维度灾难的困扰，因此在大型和复杂环境中不具备良好的可扩展性。另一方面，分散式规划使智能体能够在部分可观察环境中进行实时路径规划，展示了隐式的协调能力。然而，在密集环境中它们的收敛速度较慢且性能下降。本文介绍了一种名为CRAMP的众包感知分散式方法，通过增强式课程引导的强化学习来解决这个问题。

    Multi-Agent Path Finding (MAPF) in crowded environments presents a challenging problem in motion planning, aiming to find collision-free paths for all agents in the system. MAPF finds a wide range of applications in various domains, including aerial swarms, autonomous warehouse robotics, and self-driving vehicles. The current approaches for MAPF can be broadly categorized into two main categories: centralized and decentralized planning. Centralized planning suffers from the curse of dimensionality and thus does not scale well in large and complex environments. On the other hand, decentralized planning enables agents to engage in real-time path planning within a partially observable environment, demonstrating implicit coordination. However, they suffer from slow convergence and performance degradation in dense environments. In this paper, we introduce CRAMP, a crowd-aware decentralized approach to address this problem by leveraging reinforcement learning guided by a boosted curriculum-b
    

