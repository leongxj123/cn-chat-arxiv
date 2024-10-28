# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Measures of Resilience to Cyber Contagion -- An Axiomatic Approach for Complex Systems](https://rss.arxiv.org/abs/2312.13884) | 通过基于公理的方法，我们引入了一种新颖的风险度量方法，旨在管理复杂系统中的系统性风险，特别是网络中的系统性网络风险。这些风险度量方法针对网络的拓扑配置，以减轻传播威胁的风险。 |
| [^2] | [Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects.](http://arxiv.org/abs/2208.04883) | 本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。 |

# 详细

[^1]: 对于复杂系统的网络的弹性度量 -- 基于公理的方法

    Measures of Resilience to Cyber Contagion -- An Axiomatic Approach for Complex Systems

    [https://rss.arxiv.org/abs/2312.13884](https://rss.arxiv.org/abs/2312.13884)

    通过基于公理的方法，我们引入了一种新颖的风险度量方法，旨在管理复杂系统中的系统性风险，特别是网络中的系统性网络风险。这些风险度量方法针对网络的拓扑配置，以减轻传播威胁的风险。

    

    我们引入了一种新颖的风险度量方法，旨在管理网络中的系统性风险。与现有方法相比，这些风险度量方法针对网络的拓扑配置，以减轻传播威胁的风险。虽然我们的讨论主要围绕数字网络中系统性网络风险的管理，但我们同时将类比方法应用于其他复杂系统的风险管理，以确定是否适当。

    We introduce a novel class of risk measures designed for the management of systemic risk in networks. In contrast to prevailing approaches, these risk measures target the topological configuration of the network in order to mitigate the propagation risk of contagious threats. While our discussion primarily revolves around the management of systemic cyber risks in digital networks, we concurrently draw parallels to risk management of other complex systems where analogous approaches may be adequate.
    
[^2]: 神经会合：面向星际物体的可靠导航和控制的证明

    Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects. (arXiv:2208.04883v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2208.04883](http://arxiv.org/abs/2208.04883)

    本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。

    

    星际物体（ISOs）很可能是不可替代的原始材料，在理解系外行星星系方面具有重要价值。然而，由于其运行轨道难以约束，通常具有较高的倾角和相对速度，使用传统的人在环路方法探索ISOs具有相当大的挑战性。本文提出了一种名为神经会合的深度学习导航和控制框架，用于在实时中以可靠、准确和自主的方式遭遇快速移动的物体，包括ISOs。它在基于谱归一化的深度神经网络的引导策略之上使用点最小范数追踪控制，其中参数通过直接惩罚MPC状态轨迹跟踪误差的损失函数进行调优。我们展示了神经会合在预期的飞行器交付误差上提供了高概率指数上界，其证明利用了随机递增稳定性分析。

    Interstellar objects (ISOs) are likely representatives of primitive materials invaluable in understanding exoplanetary star systems. Due to their poorly constrained orbits with generally high inclinations and relative velocities, however, exploring ISOs with conventional human-in-the-loop approaches is significantly challenging. This paper presents Neural-Rendezvous, a deep learning-based guidance and control framework for encountering fast-moving objects, including ISOs, robustly, accurately, and autonomously in real time. It uses pointwise minimum norm tracking control on top of a guidance policy modeled by a spectrally-normalized deep neural network, where its hyperparameters are tuned with a loss function directly penalizing the MPC state trajectory tracking error. We show that Neural-Rendezvous provides a high probability exponential bound on the expected spacecraft delivery error, the proof of which leverages stochastic incremental stability analysis. In particular, it is used to
    

