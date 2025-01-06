# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Communication-Efficient Split Learning via Adaptive Feature-Wise Compression.](http://arxiv.org/abs/2307.10805) | 该论文提出了一个名为SplitFC的通信高效的分割学习框架，通过两种自适应压缩策略来减少中间特征和梯度向量的通信开销，这些策略分别是自适应特征逐渐掉落和自适应特征逐渐量化。 |

# 详细

[^1]: 通过自适应特征逐渐压缩实现高效的分割学习

    Communication-Efficient Split Learning via Adaptive Feature-Wise Compression. (arXiv:2307.10805v1 [cs.DC])

    [http://arxiv.org/abs/2307.10805](http://arxiv.org/abs/2307.10805)

    该论文提出了一个名为SplitFC的通信高效的分割学习框架，通过两种自适应压缩策略来减少中间特征和梯度向量的通信开销，这些策略分别是自适应特征逐渐掉落和自适应特征逐渐量化。

    

    本文提出了一种名为SplitFC的新颖的通信高效的分割学习（SL）框架，它减少了在SL培训过程中传输中间特征和梯度向量所需的通信开销。SplitFC的关键思想是利用矩阵的列所展示的不同的离散程度。SplitFC整合了两种压缩策略：（i）自适应特征逐渐掉落和（ii）自适应特征逐渐量化。在第一种策略中，中间特征向量根据这些向量的标准偏差确定自适应掉落概率进行掉落。然后，由于链式规则，与被丢弃的特征向量相关联的中间梯度向量也会被丢弃。在第二种策略中，非丢弃的中间特征和梯度向量使用基于向量范围确定的自适应量化级别进行量化。为了尽量减小量化误差，最优量化是。

    This paper proposes a novel communication-efficient split learning (SL) framework, named SplitFC, which reduces the communication overhead required for transmitting intermediate feature and gradient vectors during the SL training process. The key idea of SplitFC is to leverage different dispersion degrees exhibited in the columns of the matrices. SplitFC incorporates two compression strategies: (i) adaptive feature-wise dropout and (ii) adaptive feature-wise quantization. In the first strategy, the intermediate feature vectors are dropped with adaptive dropout probabilities determined based on the standard deviation of these vectors. Then, by the chain rule, the intermediate gradient vectors associated with the dropped feature vectors are also dropped. In the second strategy, the non-dropped intermediate feature and gradient vectors are quantized using adaptive quantization levels determined based on the ranges of the vectors. To minimize the quantization error, the optimal quantizatio
    

