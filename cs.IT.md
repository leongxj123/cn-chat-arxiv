# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Submodel Partitioning in Hierarchical Federated Learning: Algorithm Design and Convergence Analysis.](http://arxiv.org/abs/2310.17890) | 本文提出了一种针对分层联邦学习的新方法：分层独立子模型训练（HIST）。该方法通过将全局模型划分为不相交的子模型，并在分层结构中分布，以降低边缘设备上的计算、通信和存储负担，同时节约资源。 |

# 详细

[^1]: 分层联邦学习中的子模型划分：算法设计与收敛分析

    Submodel Partitioning in Hierarchical Federated Learning: Algorithm Design and Convergence Analysis. (arXiv:2310.17890v1 [cs.LG])

    [http://arxiv.org/abs/2310.17890](http://arxiv.org/abs/2310.17890)

    本文提出了一种针对分层联邦学习的新方法：分层独立子模型训练（HIST）。该方法通过将全局模型划分为不相交的子模型，并在分层结构中分布，以降低边缘设备上的计算、通信和存储负担，同时节约资源。

    

    分层联邦学习（HFL）相较传统的“星型拓扑”架构的联邦学习具有更好的可扩展性。然而，在资源受限的物联网（IoT）设备上训练大规模模型时，HFL仍然会对边缘设备造成重大的计算、通信和存储负担。本文提出了一种新的联邦学习方法——分层独立子模型训练（HIST），旨在解决分层场景下的这些问题。HIST的关键思想是分层版本的模型划分，即在每一轮中将全局模型划分为不相交的子模型，并将它们分布在不同的细胞中，使得每个细胞只负责训练全模型的一个划分。这样每个客户端可以节省计算和存储成本，同时减轻整个分层结构中的通信负载。我们对HIST在非凸优化问题下的收敛性行为进行了特征化分析。

    Hierarchical federated learning (HFL) has demonstrated promising scalability advantages over the traditional "star-topology" architecture-based federated learning (FL). However, HFL still imposes significant computation, communication, and storage burdens on the edge, especially when training a large-scale model over resource-constrained Internet of Things (IoT) devices. In this paper, we propose hierarchical independent submodel training (HIST), a new FL methodology that aims to address these issues in hierarchical settings. The key idea behind HIST is a hierarchical version of model partitioning, where we partition the global model into disjoint submodels in each round, and distribute them across different cells, so that each cell is responsible for training only one partition of the full model. This enables each client to save computation/storage costs while alleviating the communication loads throughout the hierarchy. We characterize the convergence behavior of HIST for non-convex 
    

