# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EncodingNet: A Novel Encoding-based MAC Design for Efficient Neural Network Acceleration](https://arxiv.org/abs/2402.18595) | 提出了一种基于编码的新型数字MAC设计，通过用简单的逻辑门代替乘法器，训练特定神经网络的位置权重，实现逐位加权累积，从而提高神经网络加速的能效和计算效果。 |

# 详细

[^1]: EncodingNet: 一种用于高效神经网络加速的基于编码的新型MAC设计

    EncodingNet: A Novel Encoding-based MAC Design for Efficient Neural Network Acceleration

    [https://arxiv.org/abs/2402.18595](https://arxiv.org/abs/2402.18595)

    提出了一种基于编码的新型数字MAC设计，通过用简单的逻辑门代替乘法器，训练特定神经网络的位置权重，实现逐位加权累积，从而提高神经网络加速的能效和计算效果。

    

    arXiv:2402.18595v1 发表类型：跨  摘要：深度神经网络（DNNs）在诸如图像分类和自然语言处理等许多领域取得了巨大突破。然而，DNN的执行需要在硬件上进行大量的乘-累积（MAC）运算，从而导致大量功耗消耗。为了解决这一挑战，我们提出了一种基于编码的新型数字MAC设计。在这种新设计中，乘法器被简单的逻辑门所取代，用于将结果投影到宽比特表示中。这些比特携带各自的位置权重，可以针对特定神经网络进行训练，以增强推断精度。新乘法器的输出通过逐位加权累积进行相加，并且累积结果与现有计算平台兼容，可加速神经网络的统一或非统一量化。由于乘法函数被简单的逻辑投影所取代，导致能量效率和计算效果的增加。

    arXiv:2402.18595v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) have achieved great breakthroughs in many fields such as image classification and natural language processing. However, the execution of DNNs needs to conduct massive numbers of multiply-accumulate (MAC) operations on hardware and thus incurs a large power consumption. To address this challenge, we propose a novel digital MAC design based on encoding. In this new design, the multipliers are replaced by simple logic gates to project the results onto a wide bit representation. These bits carry individual position weights, which can be trained for specific neural networks to enhance inference accuracy. The outputs of the new multipliers are added by bit-wise weighted accumulation and the accumulation results are compatible with existing computing platforms accelerating neural networks with either uniform or non-uniform quantization. Since the multiplication function is replaced by simple logic projection, the c
    

