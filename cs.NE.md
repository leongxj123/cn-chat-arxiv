# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous Spiking Graph Neural Networks](https://arxiv.org/abs/2404.01897) | COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。 |
| [^2] | [Dynamic Spiking Graph Neural Networks.](http://arxiv.org/abs/2401.05373) | 本文提出了一个名为"动态尖峰图神经网络"（DSGNN）的框架，它将尖峰神经网络（SNNs）与图神经网络（GNNs）结合起来，以解决动态图表示学习中的复杂性和内存开销问题。DSGNN通过动态调整尖峰神经元的状态和连接权重，在传播过程中保持图结构信息的完整性。 |
| [^3] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |

# 详细

[^1]: 连续脉冲图神经网络

    Continuous Spiking Graph Neural Networks

    [https://arxiv.org/abs/2404.01897](https://arxiv.org/abs/2404.01897)

    COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。

    

    连续图神经网络（CGNNs）因引入连续动力学而引起了极大关注，能够推广现有的离散图神经网络（GNNs）。它们通常受扩散类方法启发，引入了一种新颖的传播方案，并使用常微分方程（ODE）进行分析。然而，CGNNs的实现需要大量计算能力，这使得它们难以部署在电池供电设备上。受最近脉冲神经网络（SNNs）的启发，SNNs模拟生物推理过程并提供一种节能的神经架构，我们将SNNs与CGNNs结合到一个统一框架中，命名为连续脉冲图神经网络（COS-GNN）。我们在每个时间步骤使用SNNs进行图节点表示，这些表示进一步与时间一起集成到ODE过程中，以增强信息保存和缓解...

    arXiv:2404.01897v1 Announce Type: cross  Abstract: Continuous graph neural networks (CGNNs) have garnered significant attention due to their ability to generalize existing discrete graph neural networks (GNNs) by introducing continuous dynamics. They typically draw inspiration from diffusion-based methods to introduce a novel propagation scheme, which is analyzed using ordinary differential equations (ODE). However, the implementation of CGNNs requires significant computational power, making them challenging to deploy on battery-powered devices. Inspired by recent spiking neural networks (SNNs), which emulate a biological inference process and provide an energy-efficient neural architecture, we incorporate the SNNs with CGNNs in a unified framework, named Continuous Spiking Graph Neural Networks (COS-GNN). We employ SNNs for graph node representation at each time step, which are further integrated into the ODE process along with time. To enhance information preservation and mitigate in
    
[^2]: 动态尖峰图神经网络

    Dynamic Spiking Graph Neural Networks. (arXiv:2401.05373v1 [cs.NE])

    [http://arxiv.org/abs/2401.05373](http://arxiv.org/abs/2401.05373)

    本文提出了一个名为"动态尖峰图神经网络"（DSGNN）的框架，它将尖峰神经网络（SNNs）与图神经网络（GNNs）结合起来，以解决动态图表示学习中的复杂性和内存开销问题。DSGNN通过动态调整尖峰神经元的状态和连接权重，在传播过程中保持图结构信息的完整性。

    

    将尖峰神经网络（SNNs）和图神经网络（GNNs）相结合渐渐引起了人们的关注，这是因为它在处理由图表示的非欧几里得数据时具有低功耗和高效率。然而，作为一个常见的问题，动态图表示学习面临着高复杂性和大内存开销的挑战。目前的工作通常通过使用二进制特征而不是连续特征的SNNs来替代循环神经网络（RNNs）进行高效训练，这会忽视图结构信息并在传播过程中导致细节的丢失。此外，优化动态尖峰模型通常需要在时间步之间传播信息，这增加了内存需求。为了解决这些挑战，我们提出了一个名为"动态尖峰图神经网络"（\method{}）的框架。为了减轻信息丢失问题，\method{} 在传播过程中引入了一种新的机制，它在每个时间步骤中动态地调整尖峰神经元的状态和连接权重，以保持图结构信息的完整性。

    The integration of Spiking Neural Networks (SNNs) and Graph Neural Networks (GNNs) is gradually attracting attention due to the low power consumption and high efficiency in processing the non-Euclidean data represented by graphs. However, as a common problem, dynamic graph representation learning faces challenges such as high complexity and large memory overheads. Current work often uses SNNs instead of Recurrent Neural Networks (RNNs) by using binary features instead of continuous ones for efficient training, which would overlooks graph structure information and leads to the loss of details during propagation. Additionally, optimizing dynamic spiking models typically requires propagation of information across time steps, which increases memory requirements. To address these challenges, we present a framework named \underline{Dy}namic \underline{S}p\underline{i}king \underline{G}raph \underline{N}eural Networks (\method{}). To mitigate the information loss problem, \method{} propagates
    
[^3]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    

