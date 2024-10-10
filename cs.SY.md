# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A neural network-based approach to hybrid systems identification for control](https://arxiv.org/abs/2404.01814) | 通过神经网络架构设计出的混合系统模型具有分段线性动力学，可以用于优化控制设计，并且在有限视野最优控制问题中计算出具有强局部最优性保证的最优解。 |
| [^2] | [Networked Multiagent Reinforcement Learning for Peer-to-Peer Energy Trading.](http://arxiv.org/abs/2401.13947) | 本文提出了一个利用多智能体强化学习框架来实现点对点能源交易的方法，该方法帮助自动化消费者的竞标和管理，并解决了可再生能源零边际成本和物理约束的问题。 |
| [^3] | [Safety Margins for Reinforcement Learning.](http://arxiv.org/abs/2307.13642) | 本论文提出了一种能够通过计算代理关键性指标来生成安全边界的方法，该方法能够将可能的错误行为的后果与整体性能的预期损失联系起来。在Atari环境中的实验结果表明，随着代理接近失败状态，安全边界减小。 |
| [^4] | [Approximate non-linear model predictive control with safety-augmented neural networks.](http://arxiv.org/abs/2304.09575) | 本文提出了一种基于神经网络(NNs)的非线性模型预测控制(MPC)的近似方法，称为安全增强，可以使解决方案在线可行并具有收敛和约束条件的确定保证。 |

# 详细

[^1]: 一种基于神经网络的混合系统识别控制方法

    A neural network-based approach to hybrid systems identification for control

    [https://arxiv.org/abs/2404.01814](https://arxiv.org/abs/2404.01814)

    通过神经网络架构设计出的混合系统模型具有分段线性动力学，可以用于优化控制设计，并且在有限视野最优控制问题中计算出具有强局部最优性保证的最优解。

    

    我们考虑了设计一种基于机器学习的模型的问题，该模型可以从有限数量的(状态-输入)-后继状态数据点中识别未知动态系统，并且该模型适用于优化控制设计。我们提出了一种特定的神经网络(NN)架构，该架构产生具有分段线性动力学的混合系统，该系统对网络参数具有可微性，从而使得可以使用基于导数的训练过程。我们展示了我们的NN权重的精心选择可以产生具有非常有利结构属性的混合系统模型，当作为有限视野最优控制问题(OCP)的一部分使用时，具有很强的局部最优性保证。具体而言，我们展示了可以通过非线性规划计算具有强局部最优性保证的最优解，与通常需要混合整数优化的一般混合系统的经典OCP相比。另外，这些模型还可以被用于故障检测和故障处理。

    arXiv:2404.01814v1 Announce Type: cross  Abstract: We consider the problem of designing a machine learning-based model of an unknown dynamical system from a finite number of (state-input)-successor state data points, such that the model obtained is also suitable for optimal control design. We propose a specific neural network (NN) architecture that yields a hybrid system with piecewise-affine dynamics that is differentiable with respect to the network's parameters, thereby enabling the use of derivative-based training procedures. We show that a careful choice of our NN's weights produces a hybrid system model with structural properties that are highly favourable when used as part of a finite horizon optimal control problem (OCP). Specifically, we show that optimal solutions with strong local optimality guarantees can be computed via nonlinear programming, in contrast to classical OCPs for general hybrid systems which typically require mixed-integer optimization. In addition to being we
    
[^2]: 网络化多智能体强化学习用于点对点能源交易

    Networked Multiagent Reinforcement Learning for Peer-to-Peer Energy Trading. (arXiv:2401.13947v1 [eess.SY])

    [http://arxiv.org/abs/2401.13947](http://arxiv.org/abs/2401.13947)

    本文提出了一个利用多智能体强化学习框架来实现点对点能源交易的方法，该方法帮助自动化消费者的竞标和管理，并解决了可再生能源零边际成本和物理约束的问题。

    

    利用分布式可再生能源和能量储存资源进行点对点能源交易被长期认为是提高能源系统弹性和可持续性的解决方案。然而，消费者和自给自足者（具有能源发电资源的人）缺乏进行重复点对点交易的专业知识，并且可再生能源的零边际成本在确定公平市场价格方面存在挑战。为了解决这些问题，我们提出了多智能体强化学习（MARL）框架，以帮助自动化消费者对太阳能光伏和能量储存资源的竞标和管理，在一种利用供需比的点对点清算机制下。此外，我们展示了MARL框架如何整合物理网络约束以实现电压控制，从而确保点对点能源交易的物理可行性，并为真实世界的实施铺平了道路。

    Utilizing distributed renewable and energy storage resources in local distribution networks via peer-to-peer (P2P) energy trading has long been touted as a solution to improve energy systems' resilience and sustainability. Consumers and prosumers (those who have energy generation resources), however, do not have the expertise to engage in repeated P2P trading, and the zero-marginal costs of renewables present challenges in determining fair market prices. To address these issues, we propose multi-agent reinforcement learning (MARL) frameworks to help automate consumers' bidding and management of their solar PV and energy storage resources, under a specific P2P clearing mechanism that utilizes the so-called supply-demand ratio. In addition, we show how the MARL frameworks can integrate physical network constraints to realize voltage control, hence ensuring physical feasibility of the P2P energy trading and paving way for real-world implementations.
    
[^3]: 强化学习的安全边界

    Safety Margins for Reinforcement Learning. (arXiv:2307.13642v1 [cs.LG])

    [http://arxiv.org/abs/2307.13642](http://arxiv.org/abs/2307.13642)

    本论文提出了一种能够通过计算代理关键性指标来生成安全边界的方法，该方法能够将可能的错误行为的后果与整体性能的预期损失联系起来。在Atari环境中的实验结果表明，随着代理接近失败状态，安全边界减小。

    

    任何自主控制器在某些情况下都可能不安全。能够定量地确定何时会发生这些不安全情况对于及时引入人类监督至关重要，例如货运应用。在这项工作中，我们展示了一个代理的情况的真正关键性可以被稳健地定义为在一些随机动作下奖励的平均减少。可以将实时可计算的代理关键性指标（即，无需实际模拟随机动作的影响）与真正的关键性进行比较，并展示如何利用这些代理指标生成安全边界，将潜在错误行为的后果直接与整体性能的预期损失联系起来。我们在Atari环境中通过APE-X和A3C的学习策略上评估了我们的方法，并展示了随着代理接近失败状态，安全边界的减小。将安全边界整合到监控程序中的创新点在于...

    Any autonomous controller will be unsafe in some situations. The ability to quantitatively identify when these unsafe situations are about to occur is crucial for drawing timely human oversight in, e.g., freight transportation applications. In this work, we demonstrate that the true criticality of an agent's situation can be robustly defined as the mean reduction in reward given some number of random actions. Proxy criticality metrics that are computable in real-time (i.e., without actually simulating the effects of random actions) can be compared to the true criticality, and we show how to leverage these proxy metrics to generate safety margins, which directly tie the consequences of potentially incorrect actions to an anticipated loss in overall performance. We evaluate our approach on learned policies from APE-X and A3C within an Atari environment, and demonstrate how safety margins decrease as agents approach failure states. The integration of safety margins into programs for monit
    
[^4]: 基于安全增强神经网络的非线性近似模型预测控制

    Approximate non-linear model predictive control with safety-augmented neural networks. (arXiv:2304.09575v1 [eess.SY])

    [http://arxiv.org/abs/2304.09575](http://arxiv.org/abs/2304.09575)

    本文提出了一种基于神经网络(NNs)的非线性模型预测控制(MPC)的近似方法，称为安全增强，可以使解决方案在线可行并具有收敛和约束条件的确定保证。

    

    模型预测控制(MPC)可以实现对于一般非线性系统的稳定性和约束条件的满足，但需要进行计算开销很大的在线优化。本文研究了通过神经网络(NNs)对这种MPC控制器的近似，以实现快速的在线评估。我们提出了安全增强，尽管存在近似不准确性，但可以获得收敛和约束条件的确定保证。我们使用NN近似MPC的整个输入序列，这使得我们在线验证它是否是MPC问题的可行解。当该解决方案不可行或成本更高时，我们基于标准MPC技术将NN解决方案替换为安全候选解。我们的方法仅需要对NN进行一次评估和对输入序列进行在线前向积分，这在资源受限系统上的计算速度很快。所提出的控制框架在三个不同复杂度的非线性MPC基准上进行了演示，展示了计算效率。

    Model predictive control (MPC) achieves stability and constraint satisfaction for general nonlinear systems, but requires computationally expensive online optimization. This paper studies approximations of such MPC controllers via neural networks (NNs) to achieve fast online evaluation. We propose safety augmentation that yields deterministic guarantees for convergence and constraint satisfaction despite approximation inaccuracies. We approximate the entire input sequence of the MPC with NNs, which allows us to verify online if it is a feasible solution to the MPC problem. We replace the NN solution by a safe candidate based on standard MPC techniques whenever it is infeasible or has worse cost. Our method requires a single evaluation of the NN and forward integration of the input sequence online, which is fast to compute on resource-constrained systems. The proposed control framework is illustrated on three non-linear MPC benchmarks of different complexity, demonstrating computational
    

