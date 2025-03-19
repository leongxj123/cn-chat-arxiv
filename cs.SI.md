# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design](https://arxiv.org/abs/2310.07684) | 本文通过消息传递机制的视角，提出了一种新的对高阶网络同質性的概念化，并探索了一些处理高阶结构的策略，为超图神经网络的架构设计和性能提供了新的视野和方法。 |

# 详细

[^1]: 透过消息传递的视角看超图神经网络：同質性与架构设计的共同视野

    Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design

    [https://arxiv.org/abs/2310.07684](https://arxiv.org/abs/2310.07684)

    本文通过消息传递机制的视角，提出了一种新的对高阶网络同質性的概念化，并探索了一些处理高阶结构的策略，为超图神经网络的架构设计和性能提供了新的视野和方法。

    

    当前大部分的超图学习方法和基准数据集都是通过从图的类比中提升过来的，忽略了超图的特殊性。本文尝试解决一些相关的问题：Q1 同質性在超图神经网络中是否起到了关键作用？Q2 是否可以通过细致处理高阶网络的特征来改善当前的超图神经网络架构？Q3 现有数据集是否对超图神经网络提供了有意义的基准？为了解决这些问题，我们首先引入了基于消息传递机制的高阶网络同質性的新概念化，统一了高阶网络的分析和建模。此外，我们还研究了在超图神经网络中处理高阶结构的一些自然但大部分未被探索的策略，比如保留超边依赖的节点表示，或是以节点和超边共同编码的方式进行处理。

    Most of the current hypergraph learning methodologies and benchmarking datasets in the hypergraph realm are obtained by lifting procedures from their graph analogs, leading to overshadowing specific characteristics of hypergraphs. This paper attempts to confront some pending questions in that regard: Q1 Can the concept of homophily play a crucial role in Hypergraph Neural Networks (HNNs)? Q2 Is there room for improving current HNN architectures by carefully addressing specific characteristics of higher-order networks? Q3 Do existing datasets provide a meaningful benchmark for HNNs? To address them, we first introduce a novel conceptualization of homophily in higher-order networks based on a Message Passing (MP) scheme, unifying both the analytical examination and the modeling of higher-order networks. Further, we investigate some natural, yet mostly unexplored, strategies for processing higher-order structures within HNNs such as keeping hyperedge-dependent node representations, or per
    

