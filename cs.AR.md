# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Cost-Efficient FPGA Implementation of Tiny Transformer Model using Neural ODE.](http://arxiv.org/abs/2401.02721) | 本文提出了一种利用神经ODE作为骨干架构的高性价比FPGA实现微型Transformer模型。该模型相比于基于CNN的模型将参数大小减少了94.6%且保持准确性，适用于边缘计算。 |

# 详细

[^1]: 利用神经ODE的高性价比FPGA实现微型Transformer模型

    A Cost-Efficient FPGA Implementation of Tiny Transformer Model using Neural ODE. (arXiv:2401.02721v1 [cs.LG])

    [http://arxiv.org/abs/2401.02721](http://arxiv.org/abs/2401.02721)

    本文提出了一种利用神经ODE作为骨干架构的高性价比FPGA实现微型Transformer模型。该模型相比于基于CNN的模型将参数大小减少了94.6%且保持准确性，适用于边缘计算。

    

    Transformer是一种具有注意机制的新兴神经网络模型。它已经被用于各种任务，并且相比于CNN和RNN取得了良好的准确性。虽然注意机制被认为是一种通用的组件，但是许多Transformer模型与基于CNN的模型相比需要大量的参数。为了减少计算复杂性，最近提出了一种混合方法，它使用ResNet作为骨干架构，并将部分卷积层替换为MHSA（多头自注意）机制。在本文中，我们通过使用神经ODE（常微分方程）而不是ResNet作为骨干架构，显著减少了这种模型的参数大小。所提出的混合模型相比于基于CNN的模型将参数大小减少了94.6%，而且没有降低准确性。接着，我们将所提出的模型部署在一台适度规模的FPGA设备上进行边缘计算。

    Transformer is an emerging neural network model with attention mechanism. It has been adopted to various tasks and achieved a favorable accuracy compared to CNNs and RNNs. While the attention mechanism is recognized as a general-purpose component, many of the Transformer models require a significant number of parameters compared to the CNN-based ones. To mitigate the computational complexity, recently, a hybrid approach has been proposed, which uses ResNet as a backbone architecture and replaces a part of its convolution layers with an MHSA (Multi-Head Self-Attention) mechanism. In this paper, we significantly reduce the parameter size of such models by using Neural ODE (Ordinary Differential Equation) as a backbone architecture instead of ResNet. The proposed hybrid model reduces the parameter size by 94.6% compared to the CNN-based ones without degrading the accuracy. We then deploy the proposed model on a modest-sized FPGA device for edge computing. To further reduce FPGA resource u
    

