# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Performance-Oriented Control Barrier Functions Under Complex Safety Constraints and Limited Actuation.](http://arxiv.org/abs/2401.05629) | 本研究提出了一个新颖的自监督学习框架，通过构建可导函数来近似安全集合，并使用神经网络参数化控制障碍函数，以解决在复杂安全约束和有限执行能力下寻找最优CBF的挑战。 |
| [^2] | [Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects.](http://arxiv.org/abs/2208.04883) | 本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。 |

# 详细

[^1]: 在复杂安全约束和有限执行能力下学习以性能为导向的控制障碍函数

    Learning Performance-Oriented Control Barrier Functions Under Complex Safety Constraints and Limited Actuation. (arXiv:2401.05629v1 [cs.LG])

    [http://arxiv.org/abs/2401.05629](http://arxiv.org/abs/2401.05629)

    本研究提出了一个新颖的自监督学习框架，通过构建可导函数来近似安全集合，并使用神经网络参数化控制障碍函数，以解决在复杂安全约束和有限执行能力下寻找最优CBF的挑战。

    

    控制障碍函数（CBFs）提供了一个优雅的框架，通过将非线性控制系统的轨迹约束在预定义安全集合的不变子集上，设计安全过滤器。然而，找到一个同时在最大化控制不变集体积和适应复杂安全约束方面具有挑战性的CBF，尤其是在具有执行约束的高相对度的系统中，仍然是一个问题。在这项工作中，我们提出了一个新颖的自监督学习框架，全面解决了这些障碍。给定定义安全集合的多个状态约束的布尔组合，我们的方法从构建一个单一的可导函数开始，其0超级级别集合提供了安全集合的内部近似。然后，我们使用这个函数以及一个平滑的神经网络来参数化CBF候选。最后，我们设计了基于哈密顿-雅可比的训练损失函数。

    Control Barrier Functions (CBFs) provide an elegant framework for designing safety filters for nonlinear control systems by constraining their trajectories to an invariant subset of a prespecified safe set. However, the task of finding a CBF that concurrently maximizes the volume of the resulting control invariant set while accommodating complex safety constraints, particularly in high relative degree systems with actuation constraints, continues to pose a substantial challenge. In this work, we propose a novel self-supervised learning framework that holistically addresses these hurdles. Given a Boolean composition of multiple state constraints that define the safe set, our approach starts with building a single continuously differentiable function whose 0-superlevel set provides an inner approximation of the safe set. We then use this function together with a smooth neural network to parameterize the CBF candidate. Finally, we design a training loss function based on a Hamilton-Jacobi
    
[^2]: 神经会合：面向星际物体的可靠导航和控制的证明

    Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects. (arXiv:2208.04883v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2208.04883](http://arxiv.org/abs/2208.04883)

    本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。

    

    星际物体（ISOs）很可能是不可替代的原始材料，在理解系外行星星系方面具有重要价值。然而，由于其运行轨道难以约束，通常具有较高的倾角和相对速度，使用传统的人在环路方法探索ISOs具有相当大的挑战性。本文提出了一种名为神经会合的深度学习导航和控制框架，用于在实时中以可靠、准确和自主的方式遭遇快速移动的物体，包括ISOs。它在基于谱归一化的深度神经网络的引导策略之上使用点最小范数追踪控制，其中参数通过直接惩罚MPC状态轨迹跟踪误差的损失函数进行调优。我们展示了神经会合在预期的飞行器交付误差上提供了高概率指数上界，其证明利用了随机递增稳定性分析。

    Interstellar objects (ISOs) are likely representatives of primitive materials invaluable in understanding exoplanetary star systems. Due to their poorly constrained orbits with generally high inclinations and relative velocities, however, exploring ISOs with conventional human-in-the-loop approaches is significantly challenging. This paper presents Neural-Rendezvous, a deep learning-based guidance and control framework for encountering fast-moving objects, including ISOs, robustly, accurately, and autonomously in real time. It uses pointwise minimum norm tracking control on top of a guidance policy modeled by a spectrally-normalized deep neural network, where its hyperparameters are tuned with a loss function directly penalizing the MPC state trajectory tracking error. We show that Neural-Rendezvous provides a high probability exponential bound on the expected spacecraft delivery error, the proof of which leverages stochastic incremental stability analysis. In particular, it is used to
    

