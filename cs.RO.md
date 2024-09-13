# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tightly-Coupled LiDAR-IMU-Wheel Odometry with Online Calibration of a Kinematic Model for Skid-Steering Robots](https://arxiv.org/abs/2404.02515) | 提出了一种紧耦合LiDAR-IMU-轮里程计算法，使用在线校准解决滑移转向机器人在挑战性环境中的点云退化问题。 |
| [^2] | [Long and Short-Term Constraints Driven Safe Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/2403.18209) | 本文提出了一种基于长期和短期约束的新算法用于安全强化学习，在自动驾驶任务中可以同时保证车辆的短期和长期安全性。 |
| [^3] | [EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data](https://arxiv.org/abs/2403.00564) | EfficientZero V2在有限数据情况下通过一系列改进，在多个任务中超越了当前最先进水平，并且相比于通用算法DreamerV3有显著提升 |
| [^4] | [Structured Deep Neural Networks-Based Backstepping Trajectory Tracking Control for Lagrangian Systems](https://arxiv.org/abs/2403.00381) | 提出了一种基于结构化DNN的控制器，通过设计神经网络结构确保闭环稳定性，并进一步优化参数以实现改进的控制性能，同时提供了关于跟踪误差的明确上限。 |
| [^5] | [CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration.](http://arxiv.org/abs/2309.14660) | CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。 |
| [^6] | [What Matters to Enhance Traffic Rule Compliance of Imitation Learning for Automated Driving.](http://arxiv.org/abs/2309.07808) | 本文提出了一种基于惩罚的模仿学习方法P-CSG，结合语义生成传感器融合技术，以提高端到端自动驾驶的整体性能，并解决了交通规则遵守和传感器感知问题。 |
| [^7] | [Language-Conditioned Imitation Learning with Base Skill Priors under Unstructured Data.](http://arxiv.org/abs/2305.19075) | 本文提出了一种结合基础技能先验和模仿学习的基于语言条件的通用方法，在非结构化数据下，以增强算法在适应不熟悉的环境方面的泛化能力。在零-shot设置下，在模拟和真实环境中测试，提高了CALVIN基准测试的得分。 |

# 详细

[^1]: 利用在线校准运动模型的紧耦合LiDAR-IMU-轮里程计算法用于滑移转向机器人

    Tightly-Coupled LiDAR-IMU-Wheel Odometry with Online Calibration of a Kinematic Model for Skid-Steering Robots

    [https://arxiv.org/abs/2404.02515](https://arxiv.org/abs/2404.02515)

    提出了一种紧耦合LiDAR-IMU-轮里程计算法，使用在线校准解决滑移转向机器人在挑战性环境中的点云退化问题。

    

    隧道和长廊是移动机器人具有挑战性的环境，因为在这些环境中LiDAR点云会退化。为了解决点云退化问题，本研究提出了一种用于滑移转向机器人的紧耦合LiDAR-IMU-轮里程计算法，同时还使用在线校准方法。我们提出了一个完整的线性轮子里程计因子，不仅作为运动约束，还可以执行滑移转向机器人运动模型的在线校准。尽管运动模型动态变化（例如由于胎压引起的轮胎半径变化）和地形条件变化，我们的方法能够通过在线校准来解决模型误差。此外，我们的方法能够在退化环境下（如长直廊）通过校准而实现准确定位，同时LiDAR-IMU融合运作良好。此外，我们还估计了轮子里程计的不确定性（即协方差矩阵）。

    arXiv:2404.02515v1 Announce Type: cross  Abstract: Tunnels and long corridors are challenging environments for mobile robots because a LiDAR point cloud should degenerate in these environments. To tackle point cloud degeneration, this study presents a tightly-coupled LiDAR-IMU-wheel odometry algorithm with an online calibration for skid-steering robots. We propose a full linear wheel odometry factor, which not only serves as a motion constraint but also performs the online calibration of kinematic models for skid-steering robots. Despite the dynamically changing kinematic model (e.g., wheel radii changes caused by tire pressures) and terrain conditions, our method can address the model error via online calibration. Moreover, our method enables an accurate localization in cases of degenerated environments, such as long and straight corridors, by calibration while the LiDAR-IMU fusion sufficiently operates. Furthermore, we estimate the uncertainty (i.e., covariance matrix) of the wheel o
    
[^2]: 长短期约束驱动的安全强化学习用于自动驾驶

    Long and Short-Term Constraints Driven Safe Reinforcement Learning for Autonomous Driving

    [https://arxiv.org/abs/2403.18209](https://arxiv.org/abs/2403.18209)

    本文提出了一种基于长期和短期约束的新算法用于安全强化学习，在自动驾驶任务中可以同时保证车辆的短期和长期安全性。

    

    强化学习（RL）在决策任务中被广泛使用，但由于需要与环境交互，无法保证代理的安全性，这严重限制了其在自动驾驶等工业应用中的应用。本文提出了一种基于长期和短期约束（LSTC）的新算法用于安全RL。短期约束旨在确保车辆探测到的短期状态安全，而长期约束则确保整体安全性。

    arXiv:2403.18209v1 Announce Type: cross  Abstract: Reinforcement learning (RL) has been widely used in decision-making tasks, but it cannot guarantee the agent's safety in the training process due to the requirements of interaction with the environment, which seriously limits its industrial applications such as autonomous driving. Safe RL methods are developed to handle this issue by constraining the expected safety violation costs as a training objective, but they still permit unsafe state occurrence, which is unacceptable in autonomous driving tasks. Moreover, these methods are difficult to achieve a balance between the cost and return expectations, which leads to learning performance degradation for the algorithms. In this paper, we propose a novel algorithm based on the long and short-term constraints (LSTC) for safe RL. The short-term constraint aims to guarantee the short-term state safety that the vehicle explores, while the long-term constraint ensures the overall safety of the
    
[^3]: 高效Zero V2：在有限数据下掌握离散和连续控制

    EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data

    [https://arxiv.org/abs/2403.00564](https://arxiv.org/abs/2403.00564)

    EfficientZero V2在有限数据情况下通过一系列改进，在多个任务中超越了当前最先进水平，并且相比于通用算法DreamerV3有显著提升

    

    强化学习在现实世界任务中的样本效率仍然是一个关键挑战。虽然最近的算法在提高样本效率方面取得了显著进展，但没有一个能在不同领域中一直表现出优越性能。在本文中，我们介绍了EfficientZero V2，这是一个专为高效RL算法设计的通用框架。我们将EfficientZero的性能扩展到多个领域，涵盖连续和离散行动，以及视觉和低维输入。通过一系列我们提出的改进，EfficientZero V2在有限数据设置下在各种任务中大幅超越了当前的最先进水平（SOTA）。EfficientZero V2在多个基准测试中表现出明显的进步，比如Atari 100k，Proprio Control等中，在66个评估任务中有50个取得了优越的结果。

    arXiv:2403.00564v1 Announce Type: cross  Abstract: Sample efficiency remains a crucial challenge in applying Reinforcement Learning (RL) to real-world tasks. While recent algorithms have made significant strides in improving sample efficiency, none have achieved consistently superior performance across diverse domains. In this paper, we introduce EfficientZero V2, a general framework designed for sample-efficient RL algorithms. We have expanded the performance of EfficientZero to multiple domains, encompassing both continuous and discrete actions, as well as visual and low-dimensional inputs. With a series of improvements we propose, EfficientZero V2 outperforms the current state-of-the-art (SOTA) by a significant margin in diverse tasks under the limited data setting. EfficientZero V2 exhibits a notable advancement over the prevailing general algorithm, DreamerV3, achieving superior outcomes in 50 of 66 evaluated tasks across diverse benchmarks, such as Atari 100k, Proprio Control, an
    
[^4]: 基于结构化深度神经网络的拉格朗日系统反步轨迹跟踪控制

    Structured Deep Neural Networks-Based Backstepping Trajectory Tracking Control for Lagrangian Systems

    [https://arxiv.org/abs/2403.00381](https://arxiv.org/abs/2403.00381)

    提出了一种基于结构化DNN的控制器，通过设计神经网络结构确保闭环稳定性，并进一步优化参数以实现改进的控制性能，同时提供了关于跟踪误差的明确上限。

    

    深度神经网络（DNN）越来越多地被用于学习控制器，因为其出色的逼近能力。然而，它们的黑盒特性对闭环稳定性保证和性能分析构成了重要挑战。在本文中，我们引入了一种基于结构化DNN的控制器，用于采用反推技术实现拉格朗日系统的轨迹跟踪控制。通过适当设计神经网络结构，所提出的控制器可以确保任何兼容的神经网络参数实现闭环稳定性。此外，通过进一步优化神经网络参数，可以实现更好的控制性能。此外，我们提供了关于跟踪误差的明确上限，这允许我们通过适当选择控制参数来实现所需的跟踪性能。此外，当系统模型未知时，我们提出了一种改进的拉格朗日神经网络。

    arXiv:2403.00381v1 Announce Type: cross  Abstract: Deep neural networks (DNN) are increasingly being used to learn controllers due to their excellent approximation capabilities. However, their black-box nature poses significant challenges to closed-loop stability guarantees and performance analysis. In this paper, we introduce a structured DNN-based controller for the trajectory tracking control of Lagrangian systems using backing techniques. By properly designing neural network structures, the proposed controller can ensure closed-loop stability for any compatible neural network parameters. In addition, improved control performance can be achieved by further optimizing neural network parameters. Besides, we provide explicit upper bounds on tracking errors in terms of controller parameters, which allows us to achieve the desired tracking performance by properly selecting the controller parameters. Furthermore, when system models are unknown, we propose an improved Lagrangian neural net
    
[^5]: CoFiI2P: 粗到精的图像到点云注册的对应关系

    CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration. (arXiv:2309.14660v1 [cs.CV])

    [http://arxiv.org/abs/2309.14660](http://arxiv.org/abs/2309.14660)

    CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。

    

    图像到点云（I2P）注册是机器人导航和移动建图领域中的一项基础任务。现有的I2P注册方法在点到像素级别上估计对应关系，忽略了全局对齐。然而，没有来自全局约束的高级引导的I2P匹配容易收敛到局部最优解。为了解决这个问题，本文提出了一种新的I2P注册网络CoFiI2P，通过粗到精的方式提取对应关系，以得到全局最优解。首先，将图像和点云输入到一个共享编码-解码网络中进行层次化特征提取。然后，设计了一个粗到精的匹配模块，利用特征建立稳健的特征对应关系。具体来说，在粗匹配块中，采用了一种新型的I2P变换模块，从图像和点云中捕捉同质和异质的全局信息。通过判别描述子，完成粗-细特征匹配过程。最后，通过细化匹配模块进一步提升对应关系的准确性。

    Image-to-point cloud (I2P) registration is a fundamental task in the fields of robot navigation and mobile mapping. Existing I2P registration works estimate correspondences at the point-to-pixel level, neglecting the global alignment. However, I2P matching without high-level guidance from global constraints may converge to the local optimum easily. To solve the problem, this paper proposes CoFiI2P, a novel I2P registration network that extracts correspondences in a coarse-to-fine manner for the global optimal solution. First, the image and point cloud are fed into a Siamese encoder-decoder network for hierarchical feature extraction. Then, a coarse-to-fine matching module is designed to exploit features and establish resilient feature correspondences. Specifically, in the coarse matching block, a novel I2P transformer module is employed to capture the homogeneous and heterogeneous global information from image and point cloud. With the discriminate descriptors, coarse super-point-to-su
    
[^6]: 提升模仿学习用于自动驾驶的交通规则遵守的关键因素

    What Matters to Enhance Traffic Rule Compliance of Imitation Learning for Automated Driving. (arXiv:2309.07808v1 [cs.CV])

    [http://arxiv.org/abs/2309.07808](http://arxiv.org/abs/2309.07808)

    本文提出了一种基于惩罚的模仿学习方法P-CSG，结合语义生成传感器融合技术，以提高端到端自动驾驶的整体性能，并解决了交通规则遵守和传感器感知问题。

    

    最近越来越多的研究关注于全端到端的自动驾驶技术，在这种技术中，整个驾驶流程被替换为一个简单的神经网络，由于其结构简单和推理时间快，因此变得非常吸引人。尽管这种方法大大减少了驾驶流程中的组件，但其简单性也导致解释性问题和安全问题。训练得到的策略并不总是符合交通规则，同时也很难发现其错误的原因，因为缺乏中间输出。同时，传感器对于自动驾驶的安全性和可行性也至关重要，可以帮助感知复杂驾驶场景下的周围环境。本文提出了一种全新的基于惩罚的模仿学习方法P-CSG，结合语义生成传感器融合技术，以提高端到端自动驾驶的整体性能。我们对模型的性能进行了评估。

    More research attention has recently been given to end-to-end autonomous driving technologies where the entire driving pipeline is replaced with a single neural network because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the components in driving pipeline, its simplicity also leads to interpretability problems and safety issues arXiv:2003.06404. The trained policy is not always compliant with the traffic rules and it is also hard to discover the reason for the misbehavior because of the lack of intermediate outputs. Meanwhile, Sensors are also critical to autonomous driving's security and feasibility to perceive the surrounding environment under complex driving scenarios. In this paper, we proposed P-CSG, a novel penalty-based imitation learning approach with cross semantics generation sensor fusion technologies to increase the overall performance of End-to-End Autonomous Driving. We conducted an assessment of our model's perform
    
[^7]: 基于语言条件的模仿学习与基础技能先验下的非结构化数据应用

    Language-Conditioned Imitation Learning with Base Skill Priors under Unstructured Data. (arXiv:2305.19075v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2305.19075](http://arxiv.org/abs/2305.19075)

    本文提出了一种结合基础技能先验和模仿学习的基于语言条件的通用方法，在非结构化数据下，以增强算法在适应不熟悉的环境方面的泛化能力。在零-shot设置下，在模拟和真实环境中测试，提高了CALVIN基准测试的得分。

    

    在语言条件下的机器人操作越来越受到关注，旨在开发能够理解和执行复杂任务的机器人，以实现机器人根据语言指令操作物体的目标。虽然语言条件方法在熟悉的环境中处理任务表现出了令人印象深刻的能力，但在适应不熟悉的环境设置方面遇到了限制。在本文中，我们提出了一个通用的、基于语言条件的方法，结合了基础技能先验和模仿学习在非结构化数据下，以增强算法在适应不熟悉的环境方面的泛化能力。我们在模拟和真实环境中使用零-shot设置来评估我们模型的性能。在模拟环境中，所提出的方法在CALVIN基准测试方面超过了以前报告的得分，特别是在具有挑战性的零-shot多环境设置中。完成任务的平均长度为...

    The growing interest in language-conditioned robot manipulation aims to develop robots capable of understanding and executing complex tasks, with the objective of enabling robots to interpret language commands and manipulate objects accordingly. While language-conditioned approaches demonstrate impressive capabilities for addressing tasks in familiar environments, they encounter limitations in adapting to unfamiliar environment settings. In this study, we propose a general-purpose, language-conditioned approach that combines base skill priors and imitation learning under unstructured data to enhance the algorithm's generalization in adapting to unfamiliar environments. We assess our model's performance in both simulated and real-world environments using a zero-shot setting. In the simulated environment, the proposed approach surpasses previously reported scores for CALVIN benchmark, especially in the challenging Zero-Shot Multi-Environment setting. The average completed task length, in
    

