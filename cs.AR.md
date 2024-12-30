# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning](https://arxiv.org/abs/2404.00639) | 提出了基于强化学习的乘法器设计优化框架RL-MUL，利用矩阵和张量表示乘法器的压缩树，通过定制化的奖励实现区域和延迟之间的权衡，同时扩展到优化融合乘-累加（MAC）设计。 |

# 详细

[^1]: RL-MUL：使用深度强化学习进行乘法器设计优化

    RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning

    [https://arxiv.org/abs/2404.00639](https://arxiv.org/abs/2404.00639)

    提出了基于强化学习的乘法器设计优化框架RL-MUL，利用矩阵和张量表示乘法器的压缩树，通过定制化的奖励实现区域和延迟之间的权衡，同时扩展到优化融合乘-累加（MAC）设计。

    

    乘法是许多应用中的基本操作，乘法器被广泛应用于各种电路中。然而，由于设计空间巨大，优化乘法器是具有挑战性和非平凡的。在本文中，我们提出了RL-MUL，一个基于强化学习的乘法器设计优化框架。具体来说，我们利用矩阵和张量表示乘法器的压缩树，基于这一表示，卷积神经网络可以无缝地集成为代理网络。代理可以学习根据定制化的可容忍区域与延迟之间的权衡关系来优化乘法器结构。此外，RL-MUL的能力被扩展到优化融合乘-累加（MAC）设计。实验在不同位宽的乘法器上进行。结果表明，RL-MUL生成的乘法器能够超越所有基线。

    arXiv:2404.00639v1 Announce Type: cross  Abstract: Multiplication is a fundamental operation in many applications, and multipliers are widely adopted in various circuits. However, optimizing multipliers is challenging and non-trivial due to the huge design space. In this paper, we propose RL-MUL, a multiplier design optimization framework based on reinforcement learning. Specifically, we utilize matrix and tensor representations for the compressor tree of a multiplier, based on which the convolutional neural networks can be seamlessly incorporated as the agent network. The agent can learn to optimize the multiplier structure based on a Pareto-driven reward which is customized to accommodate the trade-off between area and delay. Additionally, the capability of RL-MUL is extended to optimize the fused multiply-accumulator (MAC) designs. Experiments are conducted on different bit widths of multipliers. The results demonstrate that the multipliers produced by RL-MUL can dominate all baseli
    

