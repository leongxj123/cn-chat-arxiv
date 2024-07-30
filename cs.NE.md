# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic Spiking Graph Neural Networks.](http://arxiv.org/abs/2401.05373) | 本文提出了一个名为"动态尖峰图神经网络"（DSGNN）的框架，它将尖峰神经网络（SNNs）与图神经网络（GNNs）结合起来，以解决动态图表示学习中的复杂性和内存开销问题。DSGNN通过动态调整尖峰神经元的状态和连接权重，在传播过程中保持图结构信息的完整性。 |

# 详细

[^1]: 动态尖峰图神经网络

    Dynamic Spiking Graph Neural Networks. (arXiv:2401.05373v1 [cs.NE])

    [http://arxiv.org/abs/2401.05373](http://arxiv.org/abs/2401.05373)

    本文提出了一个名为"动态尖峰图神经网络"（DSGNN）的框架，它将尖峰神经网络（SNNs）与图神经网络（GNNs）结合起来，以解决动态图表示学习中的复杂性和内存开销问题。DSGNN通过动态调整尖峰神经元的状态和连接权重，在传播过程中保持图结构信息的完整性。

    

    将尖峰神经网络（SNNs）和图神经网络（GNNs）相结合渐渐引起了人们的关注，这是因为它在处理由图表示的非欧几里得数据时具有低功耗和高效率。然而，作为一个常见的问题，动态图表示学习面临着高复杂性和大内存开销的挑战。目前的工作通常通过使用二进制特征而不是连续特征的SNNs来替代循环神经网络（RNNs）进行高效训练，这会忽视图结构信息并在传播过程中导致细节的丢失。此外，优化动态尖峰模型通常需要在时间步之间传播信息，这增加了内存需求。为了解决这些挑战，我们提出了一个名为"动态尖峰图神经网络"（\method{}）的框架。为了减轻信息丢失问题，\method{} 在传播过程中引入了一种新的机制，它在每个时间步骤中动态地调整尖峰神经元的状态和连接权重，以保持图结构信息的完整性。

    The integration of Spiking Neural Networks (SNNs) and Graph Neural Networks (GNNs) is gradually attracting attention due to the low power consumption and high efficiency in processing the non-Euclidean data represented by graphs. However, as a common problem, dynamic graph representation learning faces challenges such as high complexity and large memory overheads. Current work often uses SNNs instead of Recurrent Neural Networks (RNNs) by using binary features instead of continuous ones for efficient training, which would overlooks graph structure information and leads to the loss of details during propagation. Additionally, optimizing dynamic spiking models typically requires propagation of information across time steps, which increases memory requirements. To address these challenges, we present a framework named \underline{Dy}namic \underline{S}p\underline{i}king \underline{G}raph \underline{N}eural Networks (\method{}). To mitigate the information loss problem, \method{} propagates
    

