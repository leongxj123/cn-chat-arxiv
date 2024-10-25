# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network](https://arxiv.org/abs/2403.18195) | 介绍了单步组装错误校正任务和LEGO错误校正组装数据集（LEGO-ECA），提出了用于这一任务的自校正组装网络（SCANet）。 |
| [^2] | [LeTO: Learning Constrained Visuomotor Policy with Differentiable Trajectory Optimization](https://arxiv.org/abs/2401.17500) | LeTO 是一种通过可微分轨迹优化实现受限视觉运动策略的学习方法，它将优化层表示为轨迹优化问题，使模型能以安全可控的方式端到端生成动作。通过引入约束信息，实现了平衡满足约束、平滑轨迹和最小化演示误差的训练目标。在仿真和实际机器人中进行了评估，表明LeTO方法在成功率上与最先进的模仿学习方法相当。 |
| [^3] | [Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects.](http://arxiv.org/abs/2208.04883) | 本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。 |

# 详细

[^1]: 用自校正组装网络纠正LEGO组装错误

    SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network

    [https://arxiv.org/abs/2403.18195](https://arxiv.org/abs/2403.18195)

    介绍了单步组装错误校正任务和LEGO错误校正组装数据集（LEGO-ECA），提出了用于这一任务的自校正组装网络（SCANet）。

    

    在机器人学和3D视觉中，自主组装面临着重大挑战，尤其是确保组装正确性。主流方法如MEPNet目前专注于基于手动提供的图像进行组件组装。然而，这些方法在需要长期规划的任务中往往难以取得满意的结果。在同一时间，我们观察到整合自校正模块可以在一定程度上缓解这些问题。受此问题启发，我们引入了单步组装错误校正任务，其中涉及识别和纠正组件组装错误。为支持这一领域的研究，我们提出了LEGO错误校正组装数据集（LEGO-ECA），包括用于组装步骤和组装失败实例的手动图像。此外，我们提出了自校正组装网络（SCANet），这是一种新颖的方法来解决这一任务。SCANet将组装的部件视为查询，

    arXiv:2403.18195v1 Announce Type: cross  Abstract: Autonomous assembly in robotics and 3D vision presents significant challenges, particularly in ensuring assembly correctness. Presently, predominant methods such as MEPNet focus on assembling components based on manually provided images. However, these approaches often fall short in achieving satisfactory results for tasks requiring long-term planning. Concurrently, we observe that integrating a self-correction module can partially alleviate such issues. Motivated by this concern, we introduce the single-step assembly error correction task, which involves identifying and rectifying misassembled components. To support research in this area, we present the LEGO Error Correction Assembly Dataset (LEGO-ECA), comprising manual images for assembly steps and instances of assembly failures. Additionally, we propose the Self-Correct Assembly Network (SCANet), a novel method to address this task. SCANet treats assembled components as queries, de
    
[^2]: LeTO：通过可微分轨迹优化学习受限视觉运动策略

    LeTO: Learning Constrained Visuomotor Policy with Differentiable Trajectory Optimization

    [https://arxiv.org/abs/2401.17500](https://arxiv.org/abs/2401.17500)

    LeTO 是一种通过可微分轨迹优化实现受限视觉运动策略的学习方法，它将优化层表示为轨迹优化问题，使模型能以安全可控的方式端到端生成动作。通过引入约束信息，实现了平衡满足约束、平滑轨迹和最小化演示误差的训练目标。在仿真和实际机器人中进行了评估，表明LeTO方法在成功率上与最先进的模仿学习方法相当。

    

    本文介绍了一种名为LeTO的方法，通过可微分轨迹优化实现受限视觉运动策略的学习。我们的方法独特地将一个可微分优化层整合到神经网络中。通过将优化层表示为一个轨迹优化问题，我们能够使模型以安全和可控的方式端到端生成动作，而无需额外的模块。我们的方法允许在训练过程中引入约束信息，从而平衡满足约束、平滑轨迹和最小化演示误差的训练目标。这种“灰盒”方法将基于优化的安全性和可解释性与神经网络的强大表达能力结合在一起。我们在仿真和实际机器人上对LeTO进行了定量评估。在仿真中，LeTO的成功率与最先进的模仿学习方法相当，但生成的轨迹的不一致性较小。

    This paper introduces LeTO, a method for learning constrained visuomotor policy via differentiable trajectory optimization. Our approach uniquely integrates a differentiable optimization layer into the neural network. By formulating the optimization layer as a trajectory optimization problem, we enable the model to end-to-end generate actions in a safe and controlled fashion without extra modules. Our method allows for the introduction of constraints information during the training process, thereby balancing the training objectives of satisfying constraints, smoothing the trajectories, and minimizing errors with demonstrations. This "gray box" method marries the optimization-based safety and interpretability with the powerful representational abilities of neural networks. We quantitatively evaluate LeTO in simulation and on the real robot. In simulation, LeTO achieves a success rate comparable to state-of-the-art imitation learning methods, but the generated trajectories are of less un
    
[^3]: 神经会合：面向星际物体的可靠导航和控制的证明

    Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects. (arXiv:2208.04883v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2208.04883](http://arxiv.org/abs/2208.04883)

    本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。

    

    星际物体（ISOs）很可能是不可替代的原始材料，在理解系外行星星系方面具有重要价值。然而，由于其运行轨道难以约束，通常具有较高的倾角和相对速度，使用传统的人在环路方法探索ISOs具有相当大的挑战性。本文提出了一种名为神经会合的深度学习导航和控制框架，用于在实时中以可靠、准确和自主的方式遭遇快速移动的物体，包括ISOs。它在基于谱归一化的深度神经网络的引导策略之上使用点最小范数追踪控制，其中参数通过直接惩罚MPC状态轨迹跟踪误差的损失函数进行调优。我们展示了神经会合在预期的飞行器交付误差上提供了高概率指数上界，其证明利用了随机递增稳定性分析。

    Interstellar objects (ISOs) are likely representatives of primitive materials invaluable in understanding exoplanetary star systems. Due to their poorly constrained orbits with generally high inclinations and relative velocities, however, exploring ISOs with conventional human-in-the-loop approaches is significantly challenging. This paper presents Neural-Rendezvous, a deep learning-based guidance and control framework for encountering fast-moving objects, including ISOs, robustly, accurately, and autonomously in real time. It uses pointwise minimum norm tracking control on top of a guidance policy modeled by a spectrally-normalized deep neural network, where its hyperparameters are tuned with a loss function directly penalizing the MPC state trajectory tracking error. We show that Neural-Rendezvous provides a high probability exponential bound on the expected spacecraft delivery error, the proof of which leverages stochastic incremental stability analysis. In particular, it is used to
    

