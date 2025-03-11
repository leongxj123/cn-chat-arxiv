# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning](https://arxiv.org/abs/2403.15648) | SRLM 提出了一种结合了大型语言模型和深度强化学习的新型混合方法，用于人机交互式社交机器人导航，通过实时的人类语言指令推断全局规划，并在公共空间中提供多种社交服务，表现出出色的性能。 |
| [^2] | [A-KIT: Adaptive Kalman-Informed Transformer.](http://arxiv.org/abs/2401.09987) | 这项研究提出了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习传感器融合中变化的过程噪声协方差。它通过适应实际情况中的过程噪声变化，改进了估计状态的准确性，避免了滤波器发散的问题。 |
| [^3] | [Distributionally Robust Statistical Verification with Imprecise Neural Networks.](http://arxiv.org/abs/2308.14815) | 本文提出了一种使用不精确神经网络的分布鲁棒统计验证方法，通过结合主动学习、不确定性量化和神经网络验证，可以在大量的分布上提供对黑盒系统行为的保证。 |

# 详细

[^1]: SRLM: 使用大型语言模型和深度强化学习进行人机交互式社交机器人导航

    SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning

    [https://arxiv.org/abs/2403.15648](https://arxiv.org/abs/2403.15648)

    SRLM 提出了一种结合了大型语言模型和深度强化学习的新型混合方法，用于人机交互式社交机器人导航，通过实时的人类语言指令推断全局规划，并在公共空间中提供多种社交服务，表现出出色的性能。

    

    一名交互式社交机器人助手必须在复杂拥挤的空间中提供服务，根据实时的人类语言指令或反馈调整其行为。本文提出了一种名为Social Robot Planner (SRLM) 的新型混合方法，它将大型语言模型（LLM）和深度强化学习（DRL）整合起来，以在充斥着人群的公共空间中导航，并提供多种社交服务。SRLM 通过实时的人机交互指令推断全局规划，并将社交信息编码到基于LLM的大型导航模型（LNM）中，用于低层次的运动执行。此外，设计了一个基于DRL的规划器来保持基准性能，通过大型反馈模型（LFM）与LNM融合，以解决当前文本和LLM驱动的LNM的不稳定性。最后，SRLM 在广泛的实验中展示出了出色的性能。有关此工作的更多详细信息，请访问：https://sites.g

    arXiv:2403.15648v1 Announce Type: cross  Abstract: An interactive social robotic assistant must provide services in complex and crowded spaces while adapting its behavior based on real-time human language commands or feedback. In this paper, we propose a novel hybrid approach called Social Robot Planner (SRLM), which integrates Large Language Models (LLM) and Deep Reinforcement Learning (DRL) to navigate through human-filled public spaces and provide multiple social services. SRLM infers global planning from human-in-loop commands in real-time, and encodes social information into a LLM-based large navigation model (LNM) for low-level motion execution. Moreover, a DRL-based planner is designed to maintain benchmarking performance, which is blended with LNM by a large feedback model (LFM) to address the instability of current text and LLM-driven LNM. Finally, SRLM demonstrates outstanding performance in extensive experiments. More details about this work are available at: https://sites.g
    
[^2]: A-KIT:自适应Kalman-Informed Transformer

    A-KIT: Adaptive Kalman-Informed Transformer. (arXiv:2401.09987v1 [cs.RO])

    [http://arxiv.org/abs/2401.09987](http://arxiv.org/abs/2401.09987)

    这项研究提出了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习传感器融合中变化的过程噪声协方差。它通过适应实际情况中的过程噪声变化，改进了估计状态的准确性，避免了滤波器发散的问题。

    

    扩展卡尔曼滤波器(EKF)是导航应用中广泛采用的传感器融合方法。EKF的一个关键方面是在线确定反映模型不确定性的过程噪声协方差矩阵。尽管常见的EKF实现假设过程噪声是恒定的，但在实际情况中，过程噪声是变化的，导致估计状态的不准确，并可能导致滤波器发散。为了应对这种情况，提出了基于模型的自适应EKF方法，并展示了性能改进，凸显了对稳健自适应方法的需求。在本文中，我们推导并引入了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习变化的过程噪声协方差。A-KIT框架适用于任何类型的传感器融合。我们在这里介绍了基于惯性导航系统和多普勒速度日志的非线性传感器融合方法。通过使用来自自主无人潜水器的真实记录数据，我们验证了A-KIT的有效性。

    The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous und
    
[^3]: 使用不精确神经网络的分布鲁棒统计验证

    Distributionally Robust Statistical Verification with Imprecise Neural Networks. (arXiv:2308.14815v1 [cs.AI])

    [http://arxiv.org/abs/2308.14815](http://arxiv.org/abs/2308.14815)

    本文提出了一种使用不精确神经网络的分布鲁棒统计验证方法，通过结合主动学习、不确定性量化和神经网络验证，可以在大量的分布上提供对黑盒系统行为的保证。

    

    在AI安全领域，一个特别具有挑战性的问题是在高维自主系统的行为上提供保证。以可达性分析为中心的验证方法无法扩展，而纯粹的统计方法受到对采样过程的分布假设的限制。相反，我们提出了一个针对黑盒系统的分布鲁棒版本的统计验证问题，其中我们的性能保证适用于大量的分布。本文提出了一种基于主动学习、不确定性量化和神经网络验证的新方法。我们方法的一个核心部分是一种称为不精确神经网络的集成技术，它提供了不确定性以指导主动学习。主动学习使用了一种称为Sherlock的全面神经网络验证工具来收集样本。在openAI gym Mujoco环境中使用多个物理模拟器进行评估。

    A particularly challenging problem in AI safety is providing guarantees on the behavior of high-dimensional autonomous systems. Verification approaches centered around reachability analysis fail to scale, and purely statistical approaches are constrained by the distributional assumptions about the sampling process. Instead, we pose a distributionally robust version of the statistical verification problem for black-box systems, where our performance guarantees hold over a large family of distributions. This paper proposes a novel approach based on a combination of active learning, uncertainty quantification, and neural network verification. A central piece of our approach is an ensemble technique called Imprecise Neural Networks, which provides the uncertainty to guide active learning. The active learning uses an exhaustive neural-network verification tool Sherlock to collect samples. An evaluation on multiple physical simulators in the openAI gym Mujoco environments with reinforcement-
    

