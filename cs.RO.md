# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning](https://arxiv.org/abs/2403.06828) | NeuPAN 是一种实时、高度准确、无地图、适用于各种机器人且对环境不变的机器人导航解决方案，最大的创新在于将原始点直接映射到学习到的多帧距离空间，并具有端到端模型学习的可解释性，从而实现了可证明的收敛。 |

# 详细

[^1]: NeuPAN:直接点机器人导航的端到端基于模型学习

    NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning

    [https://arxiv.org/abs/2403.06828](https://arxiv.org/abs/2403.06828)

    NeuPAN 是一种实时、高度准确、无地图、适用于各种机器人且对环境不变的机器人导航解决方案，最大的创新在于将原始点直接映射到学习到的多帧距离空间，并具有端到端模型学习的可解释性，从而实现了可证明的收敛。

    

    在拥挤环境中对非全向机器人进行导航需要极其精确的感知和运动以避免碰撞。本文提出NeuPAN：一种实时、高度准确、无地图、适用于各种机器人，且对环境不变的机器人导航解决方案。NeuPAN采用紧耦合的感知-运动框架，与现有方法相比有两个关键创新：1）它直接将原始点映射到学习到的多帧距离空间，避免了从感知到控制的误差传播；2）从端到端基于模型学习的角度进行解释，实现了可证明的收敛。NeuPAN的关键在于利用插拔式（PnP）交替最小化传感器（PAN）网络解高维端到端数学模型，其中包含各种点级约束，使NeuPAN能够直接生成实时、端到端、物理可解释的运动。

    arXiv:2403.06828v1 Announce Type: cross  Abstract: Navigating a nonholonomic robot in a cluttered environment requires extremely accurate perception and locomotion for collision avoidance. This paper presents NeuPAN: a real-time, highly-accurate, map-free, robot-agnostic, and environment-invariant robot navigation solution. Leveraging a tightly-coupled perception-locomotion framework, NeuPAN has two key innovations compared to existing approaches: 1) it directly maps raw points to a learned multi-frame distance space, avoiding error propagation from perception to control; 2) it is interpretable from an end-to-end model-based learning perspective, enabling provable convergence. The crux of NeuPAN is to solve a high-dimensional end-to-end mathematical model with various point-level constraints using the plug-and-play (PnP) proximal alternating-minimization network (PAN) with neurons in the loop. This allows NeuPAN to generate real-time, end-to-end, physically-interpretable motions direct
    

