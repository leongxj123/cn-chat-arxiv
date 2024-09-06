# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey for Foundation Models in Autonomous Driving](https://rss.arxiv.org/abs/2402.01105) | 本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。 |
| [^2] | [Tightly-Coupled LiDAR-IMU-Wheel Odometry with Online Calibration of a Kinematic Model for Skid-Steering Robots](https://arxiv.org/abs/2404.02515) | 提出了一种紧耦合LiDAR-IMU-轮里程计算法，使用在线校准解决滑移转向机器人在挑战性环境中的点云退化问题。 |
| [^3] | [FRAC-Q-Learning: A Reinforcement Learning with Boredom Avoidance Processes for Social Robots](https://arxiv.org/abs/2311.15327) | FRAC-Q-Learning是一种专为社交机器人设计，能避免用户厌烦的强化学习方法，比传统算法在兴趣和厌烦程度上表现更好，有助于开发不会让用户感到无聊的社交机器人。 |
| [^4] | [MimicTouch: Learning Human's Control Strategy with Multi-Modal Tactile Feedback.](http://arxiv.org/abs/2310.16917) | MimicTouch是一种新的框架，能够模仿人类的触觉引导控制策略，通过收集来自人类示范者的多模态触觉数据集，来学习并执行复杂任务。 |
| [^5] | [Hierarchical Generative Adversarial Imitation Learning with Mid-level Input Generation for Autonomous Driving on Urban Environments.](http://arxiv.org/abs/2302.04823) | 本研究提出了一种名为hGAIL的架构，用于解决车辆的自主导航问题，通过将感知信息直接映射到低级动作的同时，学习车辆环境的中级输入表示。 |

# 详细

[^1]: 自动驾驶领域基础模型综述

    A Survey for Foundation Models in Autonomous Driving

    [https://rss.arxiv.org/abs/2402.01105](https://rss.arxiv.org/abs/2402.01105)

    本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。

    

    基于基础模型的出现，自然语言处理和计算机视觉领域发生了革命，为自动驾驶应用铺平了道路。本综述论文对40多篇研究论文进行了全面的回顾，展示了基础模型在提升自动驾驶中的作用。大型语言模型在自动驾驶的规划和仿真中发挥着重要作用，特别是通过其在推理、代码生成和翻译方面的能力。与此同时，视觉基础模型在关键任务中得到越来越广泛的应用，例如三维物体检测和跟踪，以及为仿真和测试创建逼真的驾驶场景。多模态基础模型可以整合多样的输入，展现出卓越的视觉理解和空间推理能力，对于端到端自动驾驶至关重要。本综述不仅提供了一个结构化的分类，根据模态和自动驾驶领域中的功能对基础模型进行分类，还深入研究了方法。

    The advent of foundation models has revolutionized the fields of natural language processing and computer vision, paving the way for their application in autonomous driving (AD). This survey presents a comprehensive review of more than 40 research papers, demonstrating the role of foundation models in enhancing AD. Large language models contribute to planning and simulation in AD, particularly through their proficiency in reasoning, code generation and translation. In parallel, vision foundation models are increasingly adapted for critical tasks such as 3D object detection and tracking, as well as creating realistic driving scenarios for simulation and testing. Multi-modal foundation models, integrating diverse inputs, exhibit exceptional visual understanding and spatial reasoning, crucial for end-to-end AD. This survey not only provides a structured taxonomy, categorizing foundation models based on their modalities and functionalities within the AD domain but also delves into the meth
    
[^2]: 利用在线校准运动模型的紧耦合LiDAR-IMU-轮里程计算法用于滑移转向机器人

    Tightly-Coupled LiDAR-IMU-Wheel Odometry with Online Calibration of a Kinematic Model for Skid-Steering Robots

    [https://arxiv.org/abs/2404.02515](https://arxiv.org/abs/2404.02515)

    提出了一种紧耦合LiDAR-IMU-轮里程计算法，使用在线校准解决滑移转向机器人在挑战性环境中的点云退化问题。

    

    隧道和长廊是移动机器人具有挑战性的环境，因为在这些环境中LiDAR点云会退化。为了解决点云退化问题，本研究提出了一种用于滑移转向机器人的紧耦合LiDAR-IMU-轮里程计算法，同时还使用在线校准方法。我们提出了一个完整的线性轮子里程计因子，不仅作为运动约束，还可以执行滑移转向机器人运动模型的在线校准。尽管运动模型动态变化（例如由于胎压引起的轮胎半径变化）和地形条件变化，我们的方法能够通过在线校准来解决模型误差。此外，我们的方法能够在退化环境下（如长直廊）通过校准而实现准确定位，同时LiDAR-IMU融合运作良好。此外，我们还估计了轮子里程计的不确定性（即协方差矩阵）。

    arXiv:2404.02515v1 Announce Type: cross  Abstract: Tunnels and long corridors are challenging environments for mobile robots because a LiDAR point cloud should degenerate in these environments. To tackle point cloud degeneration, this study presents a tightly-coupled LiDAR-IMU-wheel odometry algorithm with an online calibration for skid-steering robots. We propose a full linear wheel odometry factor, which not only serves as a motion constraint but also performs the online calibration of kinematic models for skid-steering robots. Despite the dynamically changing kinematic model (e.g., wheel radii changes caused by tire pressures) and terrain conditions, our method can address the model error via online calibration. Moreover, our method enables an accurate localization in cases of degenerated environments, such as long and straight corridors, by calibration while the LiDAR-IMU fusion sufficiently operates. Furthermore, we estimate the uncertainty (i.e., covariance matrix) of the wheel o
    
[^3]: FRAC-Q-Learning: 一种具有避免厌烦过程的社交机器人强化学习方法

    FRAC-Q-Learning: A Reinforcement Learning with Boredom Avoidance Processes for Social Robots

    [https://arxiv.org/abs/2311.15327](https://arxiv.org/abs/2311.15327)

    FRAC-Q-Learning是一种专为社交机器人设计，能避免用户厌烦的强化学习方法，比传统算法在兴趣和厌烦程度上表现更好，有助于开发不会让用户感到无聊的社交机器人。

    

    强化学习算法经常被应用于社交机器人。然而，大多数强化学习算法并未针对社交机器人进行优化，因此可能会让用户感到无聊。我们提出了一种专为社交机器人设计的新强化学习方法，FRAC-Q-Learning，可以避免用户感到无聊。该算法除了随机化和分类过程外，还包括一个遗忘过程。本研究通过与传统Q-Learning的比较评估了FRAC-Q-Learning的兴趣和厌烦程度分数。FRAC-Q-Learning显示出明显更高的兴趣分数趋势，并且相较于传统Q-Learning更难让用户感到无聊。因此，FRAC-Q-Learning有助于开发不会让用户感到无聊的社交机器人。该算法还可以在基于Web的通信和教育中找到应用。

    arXiv:2311.15327v3 Announce Type: replace-cross  Abstract: The reinforcement learning algorithms have often been applied to social robots. However, most reinforcement learning algorithms were not optimized for the use of social robots, and consequently they may bore users. We proposed a new reinforcement learning method specialized for the social robot, the FRAC-Q-learning, that can avoid user boredom. The proposed algorithm consists of a forgetting process in addition to randomizing and categorizing processes. This study evaluated interest and boredom hardness scores of the FRAC-Q-learning by a comparison with the traditional Q-learning. The FRAC-Q-learning showed significantly higher trend of interest score, and indicated significantly harder to bore users compared to the traditional Q-learning. Therefore, the FRAC-Q-learning can contribute to develop a social robot that will not bore users. The proposed algorithm can also find applications in Web-based communication and educational 
    
[^4]: MimicTouch: 使用多模态触觉反馈学习人类的控制策略

    MimicTouch: Learning Human's Control Strategy with Multi-Modal Tactile Feedback. (arXiv:2310.16917v1 [cs.RO])

    [http://arxiv.org/abs/2310.16917](http://arxiv.org/abs/2310.16917)

    MimicTouch是一种新的框架，能够模仿人类的触觉引导控制策略，通过收集来自人类示范者的多模态触觉数据集，来学习并执行复杂任务。

    

    在机器人技术和人工智能领域，触觉处理的整合变得越来越重要，特别是在学习执行像对准和插入这样复杂任务时。然而，现有研究主要依赖机器人遥操作数据和强化学习，忽视了人类受触觉反馈引导下的控制策略所提供的丰富见解。为了利用人类感觉，现有的从人类学习的方法主要利用视觉反馈，常常忽视了人类本能地利用触觉反馈完成复杂操作的宝贵经验。为了填补这一空白，我们引入了一种新框架"MimicTouch"，模仿人类的触觉引导控制策略。在这个框架中，我们首先从人类示范者那里收集多模态触觉数据集，包括人类触觉引导的控制策略来完成任务。接下来的步骤涉及指令的传递，其中机器人通过模仿人类的触觉引导策略来执行任务。

    In robotics and artificial intelligence, the integration of tactile processing is becoming increasingly pivotal, especially in learning to execute intricate tasks like alignment and insertion. However, existing works focusing on tactile methods for insertion tasks predominantly rely on robot teleoperation data and reinforcement learning, which do not utilize the rich insights provided by human's control strategy guided by tactile feedback. For utilizing human sensations, methodologies related to learning from humans predominantly leverage visual feedback, often overlooking the invaluable tactile feedback that humans inherently employ to finish complex manipulations. Addressing this gap, we introduce "MimicTouch", a novel framework that mimics human's tactile-guided control strategy. In this framework, we initially collect multi-modal tactile datasets from human demonstrators, incorporating human tactile-guided control strategies for task completion. The subsequent step involves instruc
    
[^5]: 基于分层生成对抗模拟学习的自动驾驶在城市环境中的应用

    Hierarchical Generative Adversarial Imitation Learning with Mid-level Input Generation for Autonomous Driving on Urban Environments. (arXiv:2302.04823v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04823](http://arxiv.org/abs/2302.04823)

    本研究提出了一种名为hGAIL的架构，用于解决车辆的自主导航问题，通过将感知信息直接映射到低级动作的同时，学习车辆环境的中级输入表示。

    

    对于现实中的城市导航场景，设计健壮的控制策略并不是一项简单的任务。在端到端的方法中，这些策略必须将车辆摄像头获得的高维图像映射到低级动作，如转向和油门。本研究提出了一种名为hGAIL的架构，用于解决车辆的自主导航问题，通过将感知信息直接映射到低级动作的同时，学习车辆环境的中级输入表示。

    Deriving robust control policies for realistic urban navigation scenarios is not a trivial task. In an end-to-end approach, these policies must map high-dimensional images from the vehicle's cameras to low-level actions such as steering and throttle. While pure Reinforcement Learning (RL) approaches are based exclusively on rewards,Generative Adversarial Imitation Learning (GAIL) agents learn from expert demonstrations while interacting with the environment, which favors GAIL on tasks for which a reward signal is difficult to derive. In this work, the hGAIL architecture was proposed to solve the autonomous navigation of a vehicle in an end-to-end approach, mapping sensory perceptions directly to low-level actions, while simultaneously learning mid-level input representations of the agent's environment. The proposed hGAIL consists of an hierarchical Adversarial Imitation Learning architecture composed of two main modules: the GAN (Generative Adversarial Nets) which generates the Bird's-
    

