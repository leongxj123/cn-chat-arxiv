# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Networks as Fast and High-fidelity Emulators for Finite-Element Ice Sheet Modeling](https://arxiv.org/abs/2402.05291) | 本研究开发了图神经网络作为有限元冰盖模拟的快速高保真度的仿真器，并在Pine Island Glacier的瞬态模拟中展示了与传统卷积神经网络和多层感知器相比更准确的重现冰盖厚度和速度的能力。同时，这些GNN成功捕捉到了更高底部熔化速率引起的冰质量损失和加速过程。在图形处理单元上实现的GNN仿真器显示出高达50倍的加速。 |

# 详细

[^1]: 图神经网络作为有限元冰盖模拟的快速高保真度的仿真器

    Graph Neural Networks as Fast and High-fidelity Emulators for Finite-Element Ice Sheet Modeling

    [https://arxiv.org/abs/2402.05291](https://arxiv.org/abs/2402.05291)

    本研究开发了图神经网络作为有限元冰盖模拟的快速高保真度的仿真器，并在Pine Island Glacier的瞬态模拟中展示了与传统卷积神经网络和多层感知器相比更准确的重现冰盖厚度和速度的能力。同时，这些GNN成功捕捉到了更高底部熔化速率引起的冰质量损失和加速过程。在图形处理单元上实现的GNN仿真器显示出高达50倍的加速。

    

    虽然冰盖和海平面系统模型（ISSM）的有限元方法可以快速准确地解决由Stokes方程描述的冰动力学问题，但这种数值建模需要在中央处理单元（CPU）上进行密集的计算。在本研究中，我们开发了图神经网络（GNN）作为快速代理模型来保持ISSM的有限元结构。利用Pine Island Glacier（PIG）的20年瞬态模拟，我们训练和测试了三个GNN：图卷积网络（GCN），图注意力网络（GAT）和等变图卷积网络（EGCN）。这些GNN与经典卷积神经网络（CNN）和多层感知器（MLP）相比，能够更准确地重现冰厚度和速度。特别是，在PIG中，GNN成功捕捉到了由更高底部熔化速率引起的冰质量损失和加速。当我们的GNN仿真器在图形处理单元（GPU）上实现时，它们显示出高达50倍的加速。

    Although the finite element approach of the Ice-sheet and Sea-level System Model (ISSM) solves ice dynamics problems governed by Stokes equations quickly and accurately, such numerical modeling requires intensive computation on central processing units (CPU). In this study, we develop graph neural networks (GNN) as fast surrogate models to preserve the finite element structure of ISSM. Using the 20-year transient simulations in the Pine Island Glacier (PIG), we train and test three GNNs: graph convolutional network (GCN), graph attention network (GAT), and equivariant graph convolutional network (EGCN). These GNNs reproduce ice thickness and velocity with better accuracy than the classic convolutional neural network (CNN) and multi-layer perception (MLP). In particular, GNNs successfully capture the ice mass loss and acceleration induced by higher basal melting rates in the PIG. When our GNN emulators are implemented on graphic processing units (GPUs), they show up to 50 times faster c
    

