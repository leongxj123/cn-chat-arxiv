# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attacking Graph Neural Networks with Bit Flips: Weisfeiler and Lehman Go Indifferent.](http://arxiv.org/abs/2311.01205) | 我们设计了Injectivity Bit Flip Attack来针对图神经网络，成功地降低了其对图结构的识别能力和表达能力，从而增加了其对位反转攻击的易受攻击性。 |

# 详细

[^1]: 使用位反转攻击图神经网络：Weisfeiler和Lehman变得冷漠了

    Attacking Graph Neural Networks with Bit Flips: Weisfeiler and Lehman Go Indifferent. (arXiv:2311.01205v1 [cs.LG])

    [http://arxiv.org/abs/2311.01205](http://arxiv.org/abs/2311.01205)

    我们设计了Injectivity Bit Flip Attack来针对图神经网络，成功地降低了其对图结构的识别能力和表达能力，从而增加了其对位反转攻击的易受攻击性。

    

    先前对图神经网络的攻击主要集中在图毒化和规避上，忽略了网络的权重和偏置。传统的基于权重的故障注入攻击，如用于卷积神经网络的位反转攻击，没有考虑到图神经网络的独特属性。我们提出了注入率位反转攻击（Injectivity Bit Flip Attack），这是第一个专门针对图神经网络设计的位反转攻击。我们的攻击针对量化消息传递神经网络中的可学习邻域聚合函数，降低了其区分图结构的能力，失去了Weisfeiler-Lehman测试的表达能力。我们的发现表明，利用特定于某些图神经网络架构的数学属性可能会显著增加其对位反转攻击的易受攻击性。注入率位反转攻击可以将各种图属性预测数据集上训练的最大表达性同构网络降级为随机输出。

    Prior attacks on graph neural networks have mostly focused on graph poisoning and evasion, neglecting the network's weights and biases. Traditional weight-based fault injection attacks, such as bit flip attacks used for convolutional neural networks, do not consider the unique properties of graph neural networks. We propose the Injectivity Bit Flip Attack, the first bit flip attack designed specifically for graph neural networks. Our attack targets the learnable neighborhood aggregation functions in quantized message passing neural networks, degrading their ability to distinguish graph structures and losing the expressivity of the Weisfeiler-Lehman test. Our findings suggest that exploiting mathematical properties specific to certain graph neural network architectures can significantly increase their vulnerability to bit flip attacks. Injectivity Bit Flip Attacks can degrade the maximal expressive Graph Isomorphism Networks trained on various graph property prediction datasets to rando
    

