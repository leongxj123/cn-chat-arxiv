# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous Spiking Graph Neural Networks](https://arxiv.org/abs/2404.01897) | COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。 |
| [^2] | [Soda: An Object-Oriented Functional Language for Specifying Human-Centered Problems.](http://arxiv.org/abs/2310.01961) | Soda是一种面向对象的功能性编程语言，用于描述以人为中心的问题，可以自然地处理质量和数量，并通过简单的定义模型化复杂要求。 |

# 详细

[^1]: 连续脉冲图神经网络

    Continuous Spiking Graph Neural Networks

    [https://arxiv.org/abs/2404.01897](https://arxiv.org/abs/2404.01897)

    COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。

    

    连续图神经网络（CGNNs）因引入连续动力学而引起了极大关注，能够推广现有的离散图神经网络（GNNs）。它们通常受扩散类方法启发，引入了一种新颖的传播方案，并使用常微分方程（ODE）进行分析。然而，CGNNs的实现需要大量计算能力，这使得它们难以部署在电池供电设备上。受最近脉冲神经网络（SNNs）的启发，SNNs模拟生物推理过程并提供一种节能的神经架构，我们将SNNs与CGNNs结合到一个统一框架中，命名为连续脉冲图神经网络（COS-GNN）。我们在每个时间步骤使用SNNs进行图节点表示，这些表示进一步与时间一起集成到ODE过程中，以增强信息保存和缓解...

    arXiv:2404.01897v1 Announce Type: cross  Abstract: Continuous graph neural networks (CGNNs) have garnered significant attention due to their ability to generalize existing discrete graph neural networks (GNNs) by introducing continuous dynamics. They typically draw inspiration from diffusion-based methods to introduce a novel propagation scheme, which is analyzed using ordinary differential equations (ODE). However, the implementation of CGNNs requires significant computational power, making them challenging to deploy on battery-powered devices. Inspired by recent spiking neural networks (SNNs), which emulate a biological inference process and provide an energy-efficient neural architecture, we incorporate the SNNs with CGNNs in a unified framework, named Continuous Spiking Graph Neural Networks (COS-GNN). We employ SNNs for graph node representation at each time step, which are further integrated into the ODE process along with time. To enhance information preservation and mitigate in
    
[^2]: Soda:一种用于描述以人为中心问题的面向对象的功能性编程语言

    Soda: An Object-Oriented Functional Language for Specifying Human-Centered Problems. (arXiv:2310.01961v1 [cs.PL])

    [http://arxiv.org/abs/2310.01961](http://arxiv.org/abs/2310.01961)

    Soda是一种面向对象的功能性编程语言，用于描述以人为中心的问题，可以自然地处理质量和数量，并通过简单的定义模型化复杂要求。

    

    我们介绍了Soda（Symbolic Objective Descriptive Analysis），一种语言，有助于以自然的方式处理质量和数量，并极大地简化了检查它们正确性的任务。我们介绍了语言的关键属性，这些属性是由对计算机系统复杂要求进行描述的设计所激发的，并解释了如何通过简单的定义来建模这些要求时必须解决这些关键属性。我们概述了一个工具，它有助于以更透明和更少出错的方式描述问题。

    We present Soda (Symbolic Objective Descriptive Analysis), a language that helps to treat qualities and quantities in a natural way and greatly simplifies the task of checking their correctness. We present key properties for the language motivated by the design of a descriptive language to encode complex requirements on computer systems, and we explain how these key properties must be addressed to model these requirements with simple definitions. We give an overview of a tool that helps to describe problems in an easy way that we consider more transparent and less error-prone.
    

