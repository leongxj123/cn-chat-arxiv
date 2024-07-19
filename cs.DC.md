# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FAX: Scalable and Differentiable Federated Primitives in JAX](https://arxiv.org/abs/2403.07128) | FAX是一个在JAX中嵌入联邦计算原语的库，支持大规模分布式计算，提供了联邦自动微分的实现，并可解释至现有的生产跨设备联邦计算系统。 |
| [^2] | [MSPipe: Efficient Temporal GNN Training via Staleness-aware Pipeline](https://arxiv.org/abs/2402.15113) | 提出了MSPipe，一个通用而高效的MTGNNs框架，实现了最大化训练吞吐量同时保持模型准确性 |
| [^3] | [Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters.](http://arxiv.org/abs/2306.14088) | 本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。 |

# 详细

[^1]: FAX: JAX中可扩展且可微分的联邦原语

    FAX: Scalable and Differentiable Federated Primitives in JAX

    [https://arxiv.org/abs/2403.07128](https://arxiv.org/abs/2403.07128)

    FAX是一个在JAX中嵌入联邦计算原语的库，支持大规模分布式计算，提供了联邦自动微分的实现，并可解释至现有的生产跨设备联邦计算系统。

    

    我们介绍了FAX，这是一个基于JAX设计的库，旨在支持数据中心和跨设备应用中的大规模分布式和联邦计算。FAX利用JAX的分片机制，实现了原生针对TPU和最先进的JAX运行时（包括Pathways）的定位。FAX将联邦计算的基本构件嵌入JAX中，带来了三个关键好处。首先，FAX的计算可以转换为XLA HLO。其次，FAX提供了联邦自动微分的完整实现，极大地简化了联邦计算的表达。最后，FAX的计算可以解释成现有的生产跨设备联邦计算系统。我们展示了FAX为数据中心中的联邦计算提供了易编程、高性能和可扩展的框架。FAX可在https://github.com/google-research/google-research/tree/master/fax 获取。

    arXiv:2403.07128v1 Announce Type: cross  Abstract: We present FAX, a JAX-based library designed to support large-scale distributed and federated computations in both data center and cross-device applications. FAX leverages JAX's sharding mechanisms to enable native targeting of TPUs and state-of-the-art JAX runtimes, including Pathways. FAX embeds building blocks for federated computations as primitives in JAX. This enables three key benefits. First, FAX computations can be translated to XLA HLO. Second, FAX provides a full implementation of federated automatic differentiation, greatly simplifying the expression of federated computations. Last, FAX computations can be interpreted out to existing production cross-device federated compute systems. We show that FAX provides an easily programmable, performant, and scalable framework for federated computations in the data center. FAX is available at https://github.com/google-research/google-research/tree/master/fax .
    
[^2]: MSPipe: 通过意识到陈旧性的管道实现高效的时间性GNN训练

    MSPipe: Efficient Temporal GNN Training via Staleness-aware Pipeline

    [https://arxiv.org/abs/2402.15113](https://arxiv.org/abs/2402.15113)

    提出了MSPipe，一个通用而高效的MTGNNs框架，实现了最大化训练吞吐量同时保持模型准确性

    

    记忆型时间性图神经网络（MTGNNs）是一类利用节点记忆模块捕获和保留长期时间依赖关系的时间性图神经网络，相对于无记忆的对应网络具有卓越的性能。然而，在MTGNNs中，为了获取最新的信息，记忆模块的迭代读取和更新过程需要遵循时间依赖关系，这引入了显著的开销并限制了训练吞吐量。现有静态GNNs的优化不适用于MTGNNs，因为两者在训练范式、模型架构和缺乏记忆模块上存在差异。此外，它们并未有效地解决时间依赖带来的挑战，使其对MTGNN训练无效。在本文中，我们提出了MSPipe，这是一个通用而高效的MTGNNs框架，可以最大化训练吞吐量同时保持模型准确性。

    arXiv:2402.15113v1 Announce Type: new  Abstract: Memory-based Temporal Graph Neural Networks (MTGNNs) are a class of temporal graph neural networks that utilize a node memory module to capture and retain long-term temporal dependencies, leading to superior performance compared to memory-less counterparts. However, the iterative reading and updating process of the memory module in MTGNNs to obtain up-to-date information needs to follow the temporal dependencies. This introduces significant overhead and limits training throughput. Existing optimizations for static GNNs are not directly applicable to MTGNNs due to differences in training paradigm, model architecture, and the absence of a memory module. Moreover, they do not effectively address the challenges posed by temporal dependencies, making them ineffective for MTGNN training. In this paper, we propose MSPipe, a general and efficient framework for MTGNNs that maximizes training throughput while maintaining model accuracy. Our design
    
[^3]: 非同质化集群下的无线联邦学习中的私有数据聚合

    Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters. (arXiv:2306.14088v1 [cs.LG])

    [http://arxiv.org/abs/2306.14088](http://arxiv.org/abs/2306.14088)

    本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。

    

    联邦学习是通过多个参与客户端私有数据的协同训练神经网络的方法。在训练神经网络的过程中，使用一种著名并广泛使用的迭代优化算法——梯度下降算法。每个客户端使用本地数据计算局部梯度并将其发送给联合器以进行聚合。客户端数据的隐私是一个主要问题。实际上，观察到局部梯度就足以泄露客户端的数据。已研究了用于应对联邦学习中隐私问题的私有聚合方案，其中所有用户都彼此连接并与联合器连接。本文考虑了一个无线系统架构，其中客户端仅通过基站连接到联合器。当需要信息论隐私时，我们推导出通信成本的基本极限，并引入和分析了一种针对这种情况量身定制的私有聚合方案。

    Federated learning collaboratively trains a neural network on privately owned data held by several participating clients. The gradient descent algorithm, a well-known and popular iterative optimization procedure, is run to train the neural network. Every client uses its local data to compute partial gradients and sends it to the federator which aggregates the results. Privacy of the clients' data is a major concern. In fact, observing the partial gradients can be enough to reveal the clients' data. Private aggregation schemes have been investigated to tackle the privacy problem in federated learning where all the users are connected to each other and to the federator. In this paper, we consider a wireless system architecture where clients are only connected to the federator via base stations. We derive fundamental limits on the communication cost when information-theoretic privacy is required, and introduce and analyze a private aggregation scheme tailored for this setting.
    

