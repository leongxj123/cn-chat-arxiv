# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Instant Complexity Reduction in CNNs using Locality-Sensitive Hashing.](http://arxiv.org/abs/2309.17211) | 该论文提出了一个名为HASTE的模块，通过使用局部敏感哈希技术，无需任何训练或精调即可实时降低卷积神经网络的计算成本，并且在压缩特征图时几乎不损失准确性。 |

# 详细

[^1]: 使用局部敏感哈希在CNN中实现即时复杂度降低

    Instant Complexity Reduction in CNNs using Locality-Sensitive Hashing. (arXiv:2309.17211v1 [cs.CV])

    [http://arxiv.org/abs/2309.17211](http://arxiv.org/abs/2309.17211)

    该论文提出了一个名为HASTE的模块，通过使用局部敏感哈希技术，无需任何训练或精调即可实时降低卷积神经网络的计算成本，并且在压缩特征图时几乎不损失准确性。

    

    为了在资源受限的设备上降低卷积神经网络（CNN）的计算成本，结构化剪枝方法已显示出有希望的结果，在不太大程度降低准确性的情况下大大减少了浮点运算（FLOPs）。然而，大多数最新的方法要求进行精调或特定的训练过程，以实现在保留准确性和降低FLOPs之间合理折衷。这引入了计算开销的额外成本，并需要可用的训练数据。为此，我们提出了HASTE（Hashing for Tractable Efficiency），它是一个无需参数和无需数据的模块，可以作为任何常规卷积模块的即插即用替代品。它能够在不需要任何训练或精调的情况下即时降低网络的测试推理成本。通过使用局部敏感哈希（LSH）来检测特征图中的冗余，我们能够大幅压缩潜在特征图而几乎不损失准确性。

    To reduce the computational cost of convolutional neural networks (CNNs) for usage on resource-constrained devices, structured pruning approaches have shown promising results, drastically reducing floating-point operations (FLOPs) without substantial drops in accuracy. However, most recent methods require fine-tuning or specific training procedures to achieve a reasonable trade-off between retained accuracy and reduction in FLOPs. This introduces additional cost in the form of computational overhead and requires training data to be available. To this end, we propose HASTE (Hashing for Tractable Efficiency), a parameter-free and data-free module that acts as a plug-and-play replacement for any regular convolution module. It instantly reduces the network's test-time inference cost without requiring any training or fine-tuning. We are able to drastically compress latent feature maps without sacrificing much accuracy by using locality-sensitive hashing (LSH) to detect redundancies in the c
    

