# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals for Naturalistic Neuroscience](https://arxiv.org/abs/2402.14213) | 通过对比学习，利用神经网络最大化相同刺激下各个个体的EEG表示的相似性，以此实现个体间共享时空脑电图表示的学习。 |
| [^2] | [Meta-Learning Strategies through Value Maximization in Neural Networks.](http://arxiv.org/abs/2310.19919) | 本文理论上研究了在神经网络中的元学习最优策略，并提出了一个学习努力的框架，可以高效地优化控制信号，从而提升学习性能。 |

# 详细

[^1]: 个体间共享脑电图时空表示的对比学习用于自然神经科学

    Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals for Naturalistic Neuroscience

    [https://arxiv.org/abs/2402.14213](https://arxiv.org/abs/2402.14213)

    通过对比学习，利用神经网络最大化相同刺激下各个个体的EEG表示的相似性，以此实现个体间共享时空脑电图表示的学习。

    

    自然刺激诱导的神经表征揭示了人类如何对日常生活中的外围刺激做出反应。理解自然刺激处理的一般神经机制的关键在于对齐各个个体的神经活动并提取个体间的共享神经表征。本研究针对脑电图（EEG）技术，该技术以其丰富的空间和时间信息而闻名，提出了一个用于个体间共享时空脑电图表示的对比学习的通用框架（CL-SSTER）。利用对比学习的表征能力，CL-SSTER利用神经网络最大化相同刺激下各个个体的EEG表示的相似性，与不同刺激的相对应。该网络采用空间和时间卷积同时学习空间和时间模式。

    arXiv:2402.14213v1 Announce Type: cross  Abstract: Neural representations induced by naturalistic stimuli offer insights into how humans respond to peripheral stimuli in daily life. The key to understanding the general neural mechanisms underlying naturalistic stimuli processing involves aligning neural activities across individuals and extracting inter-subject shared neural representations. Targeting the Electroencephalogram (EEG) technique, known for its rich spatial and temporal information, this study presents a general framework for Contrastive Learning of Shared SpatioTemporal EEG Representations across individuals (CL-SSTER). Harnessing the representational capabilities of contrastive learning, CL-SSTER utilizes a neural network to maximize the similarity of EEG representations across individuals for identical stimuli, contrasting with those for varied stimuli. The network employed spatial and temporal convolutions to simultaneously learn the spatial and temporal patterns inhere
    
[^2]: 神经网络中基于价值最大化的元学习策略

    Meta-Learning Strategies through Value Maximization in Neural Networks. (arXiv:2310.19919v1 [cs.NE])

    [http://arxiv.org/abs/2310.19919](http://arxiv.org/abs/2310.19919)

    本文理论上研究了在神经网络中的元学习最优策略，并提出了一个学习努力的框架，可以高效地优化控制信号，从而提升学习性能。

    

    生物和人工学习代理面临诸多学习选择，包括超参数选择和任务分布的各个方面，如课程。了解如何进行这些元学习选择可以提供对生物学习者的认知控制功能的规范解释，并改进工程系统。然而，由于优化整个学习过程的复杂性，目前仍然挑战着计算现代深度网络中的最优策略。在这里，我们在一个可处理的环境中从理论上研究最优策略。我们提出了一个学习努力的框架，能够在完全规范化的目标上高效地优化控制信号：在学习过程中的折现累积性能。通过使用估计梯度下降的平均动力方程，我们获得了计算的可行性，该方程适用于简单的神经网络架构。我们的框架包容了一系列元学习和自动课程学习方法，形成了统一的框架。

    Biological and artificial learning agents face numerous choices about how to learn, ranging from hyperparameter selection to aspects of task distributions like curricula. Understanding how to make these meta-learning choices could offer normative accounts of cognitive control functions in biological learners and improve engineered systems. Yet optimal strategies remain challenging to compute in modern deep networks due to the complexity of optimizing through the entire learning process. Here we theoretically investigate optimal strategies in a tractable setting. We present a learning effort framework capable of efficiently optimizing control signals on a fully normative objective: discounted cumulative performance throughout learning. We obtain computational tractability by using average dynamical equations for gradient descent, available for simple neural network architectures. Our framework accommodates a range of meta-learning and automatic curriculum learning methods in a unified n
    

