# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving](https://arxiv.org/abs/2403.17805) | 提出了MATS-Gym，一个用于在CARLA中训练智能体的多智能体交通场景框架，能够自动生成具有可变智能体数量的交通场景并整合了各种现有的交通场景描述方法。 |
| [^2] | [Fast Ergodic Search with Kernel Functions](https://arxiv.org/abs/2403.01536) | 提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。 |
| [^3] | [MimicTouch: Learning Human's Control Strategy with Multi-Modal Tactile Feedback.](http://arxiv.org/abs/2310.16917) | MimicTouch是一种新的框架，能够模仿人类的触觉引导控制策略，通过收集来自人类示范者的多模态触觉数据集，来学习并执行复杂任务。 |

# 详细

[^1]: 多智能体自主驾驶场景驱动的课程生成

    Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving

    [https://arxiv.org/abs/2403.17805](https://arxiv.org/abs/2403.17805)

    提出了MATS-Gym，一个用于在CARLA中训练智能体的多智能体交通场景框架，能够自动生成具有可变智能体数量的交通场景并整合了各种现有的交通场景描述方法。

    

    多样化和复杂训练场景的自动化生成在许多复杂学习任务中是重要的。特别是在现实世界的应用领域，如自主驾驶，自动生成课程被认为对获得强健和通用策略至关重要。然而，在充满挑战的仿真环境中，为交通场景中的多个异构智能体进行设计通常被认为是一项繁琐且耗时的任务。在我们的工作中，我们引入了MATS-Gym，一个用于在高保真驾驶模拟器CARLA中训练智能体的多智能体交通场景框架。MATS-Gym是一个用于自主驾驶的多智能体训练框架，使用部分场景规范生成具有可变智能体数量的交通场景。这篇论文将各种现有的交通场景描述方法统一到一个单一的训练框架中，并演示了如何将其与其他自主驾驶算法集成。

    arXiv:2403.17805v1 Announce Type: cross  Abstract: The automated generation of diverse and complex training scenarios has been an important ingredient in many complex learning tasks. Especially in real-world application domains, such as autonomous driving, auto-curriculum generation is considered vital for obtaining robust and general policies. However, crafting traffic scenarios with multiple, heterogeneous agents is typically considered as a tedious and time-consuming task, especially in more complex simulation environments. In our work, we introduce MATS-Gym, a Multi-Agent Traffic Scenario framework to train agents in CARLA, a high-fidelity driving simulator. MATS-Gym is a multi-agent training framework for autonomous driving that uses partial scenario specifications to generate traffic scenarios with variable numbers of agents. This paper unifies various existing approaches to traffic scenario description into a single training framework and demonstrates how it can be integrated wi
    
[^2]: 使用核函数的快速遍历搜索

    Fast Ergodic Search with Kernel Functions

    [https://arxiv.org/abs/2403.01536](https://arxiv.org/abs/2403.01536)

    提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。

    

    遍历搜索使得对信息分布进行最佳探索成为可能，同时保证了对搜索空间的渐近覆盖。然而，当前的方法通常在搜索空间维度上具有指数计算复杂度，并且局限于欧几里得空间。我们引入了一种计算高效的遍历搜索方法。我们的贡献是双重的。首先，我们开发了基于核的遍历度量，并将其从欧几里得空间推广到李群上。我们正式证明了所建议的度量与标准遍历度量一致，同时保证了在搜索空间维度上具有线性复杂度。其次，我们推导了非线性系统的核遍历度量的一阶最优性条件，这使得轨迹优化变得更加高效。全面的数值基准测试表明，所提出的方法至少比现有最先进的算法快两个数量级。

    arXiv:2403.01536v1 Announce Type: cross  Abstract: Ergodic search enables optimal exploration of an information distribution while guaranteeing the asymptotic coverage of the search space. However, current methods typically have exponential computation complexity in the search space dimension and are restricted to Euclidean space. We introduce a computationally efficient ergodic search method. Our contributions are two-fold. First, we develop a kernel-based ergodic metric and generalize it from Euclidean space to Lie groups. We formally prove the proposed metric is consistent with the standard ergodic metric while guaranteeing linear complexity in the search space dimension. Secondly, we derive the first-order optimality condition of the kernel ergodic metric for nonlinear systems, which enables efficient trajectory optimization. Comprehensive numerical benchmarks show that the proposed method is at least two orders of magnitude faster than the state-of-the-art algorithm. Finally, we d
    
[^3]: MimicTouch: 使用多模态触觉反馈学习人类的控制策略

    MimicTouch: Learning Human's Control Strategy with Multi-Modal Tactile Feedback. (arXiv:2310.16917v1 [cs.RO])

    [http://arxiv.org/abs/2310.16917](http://arxiv.org/abs/2310.16917)

    MimicTouch是一种新的框架，能够模仿人类的触觉引导控制策略，通过收集来自人类示范者的多模态触觉数据集，来学习并执行复杂任务。

    

    在机器人技术和人工智能领域，触觉处理的整合变得越来越重要，特别是在学习执行像对准和插入这样复杂任务时。然而，现有研究主要依赖机器人遥操作数据和强化学习，忽视了人类受触觉反馈引导下的控制策略所提供的丰富见解。为了利用人类感觉，现有的从人类学习的方法主要利用视觉反馈，常常忽视了人类本能地利用触觉反馈完成复杂操作的宝贵经验。为了填补这一空白，我们引入了一种新框架"MimicTouch"，模仿人类的触觉引导控制策略。在这个框架中，我们首先从人类示范者那里收集多模态触觉数据集，包括人类触觉引导的控制策略来完成任务。接下来的步骤涉及指令的传递，其中机器人通过模仿人类的触觉引导策略来执行任务。

    In robotics and artificial intelligence, the integration of tactile processing is becoming increasingly pivotal, especially in learning to execute intricate tasks like alignment and insertion. However, existing works focusing on tactile methods for insertion tasks predominantly rely on robot teleoperation data and reinforcement learning, which do not utilize the rich insights provided by human's control strategy guided by tactile feedback. For utilizing human sensations, methodologies related to learning from humans predominantly leverage visual feedback, often overlooking the invaluable tactile feedback that humans inherently employ to finish complex manipulations. Addressing this gap, we introduce "MimicTouch", a novel framework that mimics human's tactile-guided control strategy. In this framework, we initially collect multi-modal tactile datasets from human demonstrators, incorporating human tactile-guided control strategies for task completion. The subsequent step involves instruc
    

