# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Actor-Critic Physics-informed Neural Lyapunov Control](https://arxiv.org/abs/2403.08448) | 提出了一种新方法，通过使用祖博夫的偏微分方程（PDE）来训练神经网络控制器，以及对应的李雅普诺夫证书，以最大化区域吸引力，并尊重激励约束。 |
| [^2] | [Blackout Mitigation via Physics-guided RL.](http://arxiv.org/abs/2401.09640) | 本文设计了一种物理引导的强化学习框架，利用传输网络的潮流灵敏度因子来指导强化学习训练，实现了通过实时补救前瞻决策来减轻黑暗模式的目标。 |
| [^3] | [End-to-End Reinforcement Learning of Koopman Models for Economic Nonlinear MPC.](http://arxiv.org/abs/2308.01674) | 本论文提出了一种用于经济非线性MPC的Koopman模型的端到端强化学习方法，旨在实现控制性能和计算需求之间的平衡。 |

# 详细

[^1]: 《基于演员评论者物理信息神经李雅普诺夫控制》

    Actor-Critic Physics-informed Neural Lyapunov Control

    [https://arxiv.org/abs/2403.08448](https://arxiv.org/abs/2403.08448)

    提出了一种新方法，通过使用祖博夫的偏微分方程（PDE）来训练神经网络控制器，以及对应的李雅普诺夫证书，以最大化区域吸引力，并尊重激励约束。

    

    设计具有可证保证的稳定化任务控制策略是非线性控制中的一个长期问题。关键的性能指标是产生区域吸引力的大小，这基本上充当了封闭环系统对不确定性的弹性“边界”。本文提出了一种新方法，用于训练一个稳定的神经网络控制器以及其对应的李雅普诺夫证书，旨在最大化产生的区域吸引力，同时尊重激励约束。我们方法的关键之处在于使用祖博夫的偏微分方程（PDE），该方程精确地表征了给定控制策略的真实区域吸引力。我们的框架遵循演员评论者模式，我们在改进控制策略（演员）和学习祖博夫函数（评论者）之间交替进行。最后，我们通过调用SMT求解器计算出最大的可证区域吸引力。

    arXiv:2403.08448v1 Announce Type: new  Abstract: Designing control policies for stabilization tasks with provable guarantees is a long-standing problem in nonlinear control. A crucial performance metric is the size of the resulting region of attraction, which essentially serves as a robustness "margin" of the closed-loop system against uncertainties. In this paper, we propose a new method to train a stabilizing neural network controller along with its corresponding Lyapunov certificate, aiming to maximize the resulting region of attraction while respecting the actuation constraints. Crucial to our approach is the use of Zubov's Partial Differential Equation (PDE), which precisely characterizes the true region of attraction of a given control policy. Our framework follows an actor-critic pattern where we alternate between improving the control policy (actor) and learning a Zubov function (critic). Finally, we compute the largest certifiable region of attraction by invoking an SMT solver
    
[^2]: 通过物理引导的强化学习进行停电减轻

    Blackout Mitigation via Physics-guided RL. (arXiv:2401.09640v1 [eess.SY])

    [http://arxiv.org/abs/2401.09640](http://arxiv.org/abs/2401.09640)

    本文设计了一种物理引导的强化学习框架，利用传输网络的潮流灵敏度因子来指导强化学习训练，实现了通过实时补救前瞻决策来减轻黑暗模式的目标。

    

    本文考虑了为了防止黑暗模式而在系统异常时进行序列设计的补救控制行动。设计了一种物理引导的强化学习框架，用于识别在考虑系统稳定性长期影响的情况下的实时补救前瞻决策序列。本文考虑了涉及离散值传输线开关决策（线路重新连接和移除）和连续值发电机调整的控制行动空间。为了确定有效的停电减轻策略，设计了一种物理引导方法，利用与电力传输网络相关的潮流灵敏度因子来引导强化学习训练期间的探索。使用开源Grid2Op平台进行了全面的实证评估，证明了将物理信号纳入强化学习决策的显著优势，证实了所提出的物理引导方法的收益。

    This paper considers the sequential design of remedial control actions in response to system anomalies for the ultimate objective of preventing blackouts. A physics-guided reinforcement learning (RL) framework is designed to identify effective sequences of real-time remedial look-ahead decisions accounting for the long-term impact on the system's stability. The paper considers a space of control actions that involve both discrete-valued transmission line-switching decisions (line reconnections and removals) and continuous-valued generator adjustments. To identify an effective blackout mitigation policy, a physics-guided approach is designed that uses power-flow sensitivity factors associated with the power transmission network to guide the RL exploration during agent training. Comprehensive empirical evaluations using the open-source Grid2Op platform demonstrate the notable advantages of incorporating physical signals into RL decisions, establishing the gains of the proposed physics-gu
    
[^3]: 经济非线性MPC的Koopman模型的端到端强化学习

    End-to-End Reinforcement Learning of Koopman Models for Economic Nonlinear MPC. (arXiv:2308.01674v1 [cs.LG])

    [http://arxiv.org/abs/2308.01674](http://arxiv.org/abs/2308.01674)

    本论文提出了一种用于经济非线性MPC的Koopman模型的端到端强化学习方法，旨在实现控制性能和计算需求之间的平衡。

    

    （经济）非线性模型预测控制（（e）NMPC）需要在所有相关状态空间区域都具有足够准确性的动态系统模型。这些模型还必须计算成本足够低以确保实时可行性。基于数据驱动的替代机制模型可以用来减少（e）NMPC的计算负担；但是，这些模型通常通过系统辨识以在模拟样本上获得最大平均预测准确性进行训练，并作为实际（e）NMPC的一部分表现不佳。我们提出了一种用于实现最佳（e）NMPC性能的动态替代模型的端到端强化学习方法，从而得到具有控制性能和计算需求之间良好平衡的预测控制器。我们通过两个基于已建立的非线性连续搅拌反应器模型的应用来验证我们的方法。

    (Economic) nonlinear model predictive control ((e)NMPC) requires dynamic system models that are sufficiently accurate in all relevant state-space regions. These models must also be computationally cheap enough to ensure real-time tractability. Data-driven surrogate models for mechanistic models can be used to reduce the computational burden of (e)NMPC; however, such models are typically trained by system identification for maximum average prediction accuracy on simulation samples and perform suboptimally as part of actual (e)NMPC. We present a method for end-to-end reinforcement learning of dynamic surrogate models for optimal performance in (e)NMPC applications, resulting in predictive controllers that strike a favorable balance between control performance and computational demand. We validate our method on two applications derived from an established nonlinear continuous stirred-tank reactor model. We compare the controller performance to that of MPCs utilizing models trained by the 
    

