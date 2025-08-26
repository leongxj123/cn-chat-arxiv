# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias](https://arxiv.org/abs/2402.03991) | 神经网络中的权重衰减和小的类内变化与低秩偏差现象有关 |

# 详细

[^1]: 神经网络的权重衰减和类内变化小会导致低秩偏差

    Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias

    [https://arxiv.org/abs/2402.03991](https://arxiv.org/abs/2402.03991)

    神经网络中的权重衰减和小的类内变化与低秩偏差现象有关

    

    近期在深度学习领域的研究显示了一个隐含的低秩偏差现象：深度网络中的权重矩阵往往近似为低秩，在训练过程中或从已经训练好的模型中去除相对较小的奇异值可以显著减小模型大小，同时保持甚至提升模型性能。然而，大多数关于神经网络低秩偏差的理论研究都涉及到简化的线性深度网络。在本文中，我们考虑了带有非线性激活函数和权重衰减参数的通用网络，并展示了一个有趣的神经秩崩溃现象，它将训练好的网络的低秩偏差与网络的神经崩溃特性联系起来：随着权重衰减参数的增加，网络中每一层的秩呈比例递减，与前面层的隐藏空间嵌入的类内变化成反比。我们的理论发现得到了支持。

    Recent work in deep learning has shown strong empirical and theoretical evidence of an implicit low-rank bias: weight matrices in deep networks tend to be approximately low-rank and removing relatively small singular values during training or from available trained models may significantly reduce model size while maintaining or even improving model performance. However, the majority of the theoretical investigations around low-rank bias in neural networks deal with oversimplified deep linear networks. In this work, we consider general networks with nonlinear activations and the weight decay parameter, and we show the presence of an intriguing neural rank collapse phenomenon, connecting the low-rank bias of trained networks with networks' neural collapse properties: as the weight decay parameter grows, the rank of each layer in the network decreases proportionally to the within-class variability of the hidden-space embeddings of the previous layers. Our theoretical findings are supporte
    

