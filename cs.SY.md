# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provably Safe Neural Network Controllers via Differential Dynamic Logic](https://arxiv.org/abs/2402.10998) | 通过差分动态逻辑与神经网络验证相结合的VerSAILLE方法，实现了对神经网络控制系统在无限时间范围上的安全性证明。 |
| [^2] | [Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects.](http://arxiv.org/abs/2208.04883) | 本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。 |

# 详细

[^1]: 通过差分动态逻辑确保神经网络控制器的安全性

    Provably Safe Neural Network Controllers via Differential Dynamic Logic

    [https://arxiv.org/abs/2402.10998](https://arxiv.org/abs/2402.10998)

    通过差分动态逻辑与神经网络验证相结合的VerSAILLE方法，实现了对神经网络控制系统在无限时间范围上的安全性证明。

    

    虽然神经网络（NN）作为面向目标的控制器在网络物理系统中具有巨大潜力，但验证基于神经网络的控制系统（NNCS）的安全性对于实际应用NN来说面临着重大挑战，特别是当需要对无界时间范围进行安全性验证时。我们引入了VerSAILLE（通过逻辑链接包验证的可验证安全人工智能）：这是差分动态逻辑（dL）和NN验证组合的第一种方法。通过合作，我们可以利用NN验证工具的效率，同时保留dL的严谨性。我们提出了一个控制器信封的安全性证明，以证明无限时间范围上具体NNCS的安全性。VerSAILLE导致的NN验证属性通常需要非线性算术，而高效的NN验证工具仅支持线性算术。

    arXiv:2402.10998v1 Announce Type: cross  Abstract: While neural networks (NNs) have a large potential as goal-oriented controllers for Cyber-Physical Systems, verifying the safety of neural network based control systems (NNCSs) poses significant challenges for the practical use of NNs -- especially when safety is needed for unbounded time horizons. One reason for this is the intractability of NN and hybrid system analysis. We introduce VerSAILLE (Verifiably Safe AI via Logically Linked Envelopes): The first approach for the combination of differential dynamic logic (dL) and NN verification. By joining forces, we can exploit the efficiency of NN verification tools while retaining the rigor of dL. We reflect a safety proof for a controller envelope in an NN to prove the safety of concrete NNCS on an infinite-time horizon. The NN verification properties resulting from VerSAILLE typically require nonlinear arithmetic while efficient NN verification tools merely support linear arithmetic. T
    
[^2]: 神经会合：面向星际物体的可靠导航和控制的证明

    Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects. (arXiv:2208.04883v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2208.04883](http://arxiv.org/abs/2208.04883)

    本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。

    

    星际物体（ISOs）很可能是不可替代的原始材料，在理解系外行星星系方面具有重要价值。然而，由于其运行轨道难以约束，通常具有较高的倾角和相对速度，使用传统的人在环路方法探索ISOs具有相当大的挑战性。本文提出了一种名为神经会合的深度学习导航和控制框架，用于在实时中以可靠、准确和自主的方式遭遇快速移动的物体，包括ISOs。它在基于谱归一化的深度神经网络的引导策略之上使用点最小范数追踪控制，其中参数通过直接惩罚MPC状态轨迹跟踪误差的损失函数进行调优。我们展示了神经会合在预期的飞行器交付误差上提供了高概率指数上界，其证明利用了随机递增稳定性分析。

    Interstellar objects (ISOs) are likely representatives of primitive materials invaluable in understanding exoplanetary star systems. Due to their poorly constrained orbits with generally high inclinations and relative velocities, however, exploring ISOs with conventional human-in-the-loop approaches is significantly challenging. This paper presents Neural-Rendezvous, a deep learning-based guidance and control framework for encountering fast-moving objects, including ISOs, robustly, accurately, and autonomously in real time. It uses pointwise minimum norm tracking control on top of a guidance policy modeled by a spectrally-normalized deep neural network, where its hyperparameters are tuned with a loss function directly penalizing the MPC state trajectory tracking error. We show that Neural-Rendezvous provides a high probability exponential bound on the expected spacecraft delivery error, the proof of which leverages stochastic incremental stability analysis. In particular, it is used to
    

