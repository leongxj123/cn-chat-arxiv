# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning in a Safety-Embedded MDP with Trajectory Optimization.](http://arxiv.org/abs/2310.06903) | 本文介绍了一种在安全嵌入式MDP中结合轨迹优化的强化学习方法，通过将安全约束嵌入动作空间，能够有效地在最大化奖励和遵守安全约束之间取得平衡，并在挑战性任务中取得了优异性能。 |
| [^2] | [Robotic Manipulation Datasets for Offline Compositional Reinforcement Learning.](http://arxiv.org/abs/2307.07091) | 本论文提供了离线强化学习的机器人操作数据集，使用组合式强化学习生成了四个包含256个任务的数据集。每个数据集由性能不同的代理采集，包含2.56亿条转换记录。实验结果显示， |
| [^3] | [Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation.](http://arxiv.org/abs/2305.03942) | 论文介绍了一种名为HACMan的强化学习方法，用于使用点云观察进行6D非抓取式操作的物体操纵。HACMan重点关注物体中心动作表示，它包括从物体点云中选择接触位置和一组描述机器人在接触后如何移动的运动参数。在实际测试中，HACMan的表现明显优于现有基线方法。 |
| [^4] | [Constrained Reinforcement Learning using Distributional Representation for Trustworthy Quadrotor UAV Tracking Control.](http://arxiv.org/abs/2302.11694) | 提出了一种使用分布式表示进行受限强化学习的方法，用于可信四旋翼无人机的跟踪控制。通过集成分布式强化学习干扰估计器和随机模型预测控制器，能够准确识别气动效应的不确定性，实现最优的全局收敛速率和一定的亚线性收敛速率。 |

# 详细

[^1]: 在具有轨迹优化的安全嵌入式MDP中的强化学习

    Reinforcement Learning in a Safety-Embedded MDP with Trajectory Optimization. (arXiv:2310.06903v1 [cs.RO])

    [http://arxiv.org/abs/2310.06903](http://arxiv.org/abs/2310.06903)

    本文介绍了一种在安全嵌入式MDP中结合轨迹优化的强化学习方法，通过将安全约束嵌入动作空间，能够有效地在最大化奖励和遵守安全约束之间取得平衡，并在挑战性任务中取得了优异性能。

    

    安全强化学习在将强化学习算法应用于安全关键的实际应用中起着重要的作用，解决了最大化奖励和遵守安全约束之间的权衡。本文介绍了一种新颖的方法，将强化学习与轨迹优化相结合，有效地管理这种权衡。我们的方法将安全约束嵌入到修改后的马尔可夫决策过程（MDP）的动作空间中。强化学习代理通过轨迹优化器产生一系列行动，这些行动转化为安全轨迹，从而有效确保安全并提高训练稳定性。这种新颖的方法在挑战性的Safety Gym任务的性能方面表现出色，在推理过程中实现了显著更高的奖励和几乎零的安全违规。该方法在一个真实机器人任务中的应用性得到了证明，该任务涉及推动箱子穿过障碍物。

    Safe Reinforcement Learning (RL) plays an important role in applying RL algorithms to safety-critical real-world applications, addressing the trade-off between maximizing rewards and adhering to safety constraints. This work introduces a novel approach that combines RL with trajectory optimization to manage this trade-off effectively. Our approach embeds safety constraints within the action space of a modified Markov Decision Process (MDP). The RL agent produces a sequence of actions that are transformed into safe trajectories by a trajectory optimizer, thereby effectively ensuring safety and increasing training stability. This novel approach excels in its performance on challenging Safety Gym tasks, achieving significantly higher rewards and near-zero safety violations during inference. The method's real-world applicability is demonstrated through a safe and effective deployment in a real robot task of box-pushing around obstacles.
    
[^2]: 离线组合强化学习的机器人操作数据集

    Robotic Manipulation Datasets for Offline Compositional Reinforcement Learning. (arXiv:2307.07091v1 [cs.LG])

    [http://arxiv.org/abs/2307.07091](http://arxiv.org/abs/2307.07091)

    本论文提供了离线强化学习的机器人操作数据集，使用组合式强化学习生成了四个包含256个任务的数据集。每个数据集由性能不同的代理采集，包含2.56亿条转换记录。实验结果显示，

    

    离线强化学习是一种有前途的方向，可以让强化学习代理在大规模数据集上进行预训练，避免昂贵的数据收集过程的重复。为了推动该领域的发展，生成大规模数据集至关重要。组合式强化学习对于生成这样的大规模数据集尤为有吸引力，因为1）它允许从少量组件中创建多个任务，2）任务结构可以让训练好的代理通过组合相关的学习组件来解决新任务，并且3）组合维度提供了任务关联性的概念。本文提供了四个离线强化学习数据集，用于模拟机器人操作，这些数据集使用了来自CompoSuite [Mendez et al., 2022a]的256个任务。每个数据集是从一个具有不同性能等级的代理收集的，包含了2.56亿条转换记录。我们提供了训练和评估设置，以评估代理学习组合任务策略的能力。我们在每个设置上进行的基准实验表明，

    Offline reinforcement learning (RL) is a promising direction that allows RL agents to pre-train on large datasets, avoiding the recurrence of expensive data collection. To advance the field, it is crucial to generate large-scale datasets. Compositional RL is particularly appealing for generating such large datasets, since 1) it permits creating many tasks from few components, 2) the task structure may enable trained agents to solve new tasks by combining relevant learned components, and 3) the compositional dimensions provide a notion of task relatedness. This paper provides four offline RL datasets for simulated robotic manipulation created using the 256 tasks from CompoSuite [Mendez et al., 2022a]. Each dataset is collected from an agent with a different degree of performance, and consists of 256 million transitions. We provide training and evaluation settings for assessing an agent's ability to learn compositional task policies. Our benchmarking experiments on each setting show that
    
[^3]: 学习6D非抓取式操作的混合演员-评论员地图

    Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation. (arXiv:2305.03942v1 [cs.RO])

    [http://arxiv.org/abs/2305.03942](http://arxiv.org/abs/2305.03942)

    论文介绍了一种名为HACMan的强化学习方法，用于使用点云观察进行6D非抓取式操作的物体操纵。HACMan重点关注物体中心动作表示，它包括从物体点云中选择接触位置和一组描述机器人在接触后如何移动的运动参数。在实际测试中，HACMan的表现明显优于现有基线方法。

    

    在人类的灵巧性中，非抓取式操作是操作物体的重要组成部分。非抓取式操纵可以使与物体的交互更加复杂，但也在推理交互方面提出了挑战。在本文中，我们引入了一个名为HACMan的混合演员评论员地图，这是一种使用点云观察的6D非抓取式物体操作的强化学习方法。HACMan提出了一种时间抽象和空间基础的物体中心动作表示，该表示包括从物体点云中选择接触位置和一组描述机器人在接触后如何移动的运动参数。我们修改了一个现有的离线策略RL算法，以在这种混合的离散-连续动作表示学习。我们在仿真和现实世界中对HACMan进行了6D物体姿态对齐任务的评估。在最难的任务版本中，通过随机初始化物体和机器人配置，HACMan的表现优于现有的基线方法。

    Manipulating objects without grasping them is an essential component of human dexterity, referred to as non-prehensile manipulation. Non-prehensile manipulation may enable more complex interactions with the objects, but also presents challenges in reasoning about the interactions. In this work, we introduce Hybrid Actor-Critic Maps for Manipulation (HACMan), a reinforcement learning approach for 6D non-prehensile manipulation of objects using point cloud observations. HACMan proposes a temporally-abstracted and spatially-grounded object-centric action representation that consists of selecting a contact location from the object point cloud and a set of motion parameters describing how the robot will move after making contact. We modify an existing off-policy RL algorithm to learn in this hybrid discrete-continuous action representation. We evaluate HACMan on a 6D object pose alignment task in both simulation and in the real world. On the hardest version of our task, with randomized init
    
[^4]: 受限强化学习在可信四旋翼无人机跟踪控制中的应用

    Constrained Reinforcement Learning using Distributional Representation for Trustworthy Quadrotor UAV Tracking Control. (arXiv:2302.11694v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2302.11694](http://arxiv.org/abs/2302.11694)

    提出了一种使用分布式表示进行受限强化学习的方法，用于可信四旋翼无人机的跟踪控制。通过集成分布式强化学习干扰估计器和随机模型预测控制器，能够准确识别气动效应的不确定性，实现最优的全局收敛速率和一定的亚线性收敛速率。

    

    在复杂的动态环境中，同时实现四旋翼无人机的准确和可靠的跟踪控制是具有挑战性的。由于来自气动力的阻力和力矩变化是混沌的，并且难以精确识别，大多数现有的四旋翼跟踪系统将其视为传统控制方法中的简单“干扰”。本文提出了一种新颖的、可解释的轨迹跟踪器，将分布式强化学习干扰估计器与随机模型预测控制器（SMPC）相结合，用于未知的气动效应。所提出的估计器“受限分布式强化干扰估计器”（ConsDRED）准确地识别真实气动效应与估计值之间的不确定性。采用简化仿射干扰反馈进行控制参数化，以保证凸性，然后将其与SMPC相结合。我们在理论上保证ConsDRED至少实现最优的全局收敛速率和一定的亚线性收敛速率。

    Simultaneously accurate and reliable tracking control for quadrotors in complex dynamic environments is challenging. As aerodynamics derived from drag forces and moment variations are chaotic and difficult to precisely identify, most current quadrotor tracking systems treat them as simple `disturbances' in conventional control approaches. We propose a novel, interpretable trajectory tracker integrating a Distributional Reinforcement Learning disturbance estimator for unknown aerodynamic effects with a Stochastic Model Predictive Controller (SMPC). The proposed estimator `Constrained Distributional Reinforced disturbance estimator' (ConsDRED) accurately identifies uncertainties between true and estimated values of aerodynamic effects. Simplified Affine Disturbance Feedback is used for control parameterization to guarantee convexity, which we then integrate with a SMPC. We theoretically guarantee that ConsDRED achieves at least an optimal global convergence rate and a certain sublinear r
    

