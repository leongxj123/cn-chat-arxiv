# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Physics-embedded Deep Learning Framework for Cloth Simulation](https://arxiv.org/abs/2403.12820) | 该论文提出了一种基于物理的深度学习框架，可以直接编码布料模拟的物理特征，实现快速和实时模拟，并在不使用新数据训练的情况下通过测试表现出与基线的一致性。 |

# 详细

[^1]: 一种基于物理的深度学习框架用于布料模拟

    A Physics-embedded Deep Learning Framework for Cloth Simulation

    [https://arxiv.org/abs/2403.12820](https://arxiv.org/abs/2403.12820)

    该论文提出了一种基于物理的深度学习框架，可以直接编码布料模拟的物理特征，实现快速和实时模拟，并在不使用新数据训练的情况下通过测试表现出与基线的一致性。

    

    精细的布料模拟长期以来一直是计算机图形学中所期望的。为改进受力交互、碰撞处理和数值积分，提出了各种方法。深度学习有潜力实现快速和实时模拟，但常见的神经网络结构通常需要大量参数来捕获布料动力学。本文提出了一种直接编码布料模拟物理特征的物理嵌入学习框架。卷积神经网络用于表示质点-弹簧系统的空间相关性，之后设计了三个分支来学习布料物理的线性、非线性和时间导数特征。该框架还可以通过传统模拟器或子神经网络与其他外部力和碰撞处理进行集成。模型在不使用新数据进行训练的情况下，在不同的布料动画案例中进行了测试。与基线的一致性

    arXiv:2403.12820v1 Announce Type: cross  Abstract: Delicate cloth simulations have long been desired in computer graphics. Various methods were proposed to improve engaged force interactions, collision handling, and numerical integrations. Deep learning has the potential to achieve fast and real-time simulation, but common neural network structures often demand many parameters to capture cloth dynamics. This paper proposes a physics-embedded learning framework that directly encodes physical features of cloth simulation. The convolutional neural network is used to represent spatial correlations of the mass-spring system, after which three branches are designed to learn linear, nonlinear, and time derivate features of cloth physics. The framework can also integrate with other external forces and collision handling through either traditional simulators or sub neural networks. The model is tested across different cloth animation cases, without training with new data. Agreement with baselin
    

