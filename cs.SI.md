# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation.](http://arxiv.org/abs/2305.09958) | 本文章提出了一种简单有效的异质图神经网络模型SIMGA，它通过SimRank全局聚合来解决异质性节点聚合的问题，具有接近于线性的传播效率，同时具有良好的有效性和可扩展性。 |

# 详细

[^1]: SIMGA：一种简单有效的异质图神经网络结构与高效的全局聚合

    SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation. (arXiv:2305.09958v1 [cs.LG])

    [http://arxiv.org/abs/2305.09958](http://arxiv.org/abs/2305.09958)

    本文章提出了一种简单有效的异质图神经网络模型SIMGA，它通过SimRank全局聚合来解决异质性节点聚合的问题，具有接近于线性的传播效率，同时具有良好的有效性和可扩展性。

    

    图神经网络在图学习领域取得了巨大成功，但遇到异质性时会出现性能下降，即因为局部和统一聚合而导致的相邻节点不相似。现有的异质性图神经网络中，试图整合全局聚合的尝试通常需要迭代地维护和更新全图信息，对于一个具有 $n$ 个节点的图，这需要 $\mathcal{O}(n^2)$ 的计算效率，从而导致对大型图的扩展性较差。在本文中，我们提出了 SIMGA，一种将 SimRank 结构相似度测量作为全局聚合的 GNN 结构。 SIMGA 的设计简单，且在效率和有效性方面都有着有 promising 的结果。SIMGA 的简单性使其成为第一个可以实现接近于线性的 $n$ 传播效率的异质性 GNN 模型。我们从理论上证明了它的有效性，将 SimRank 视为 GNN 的一种新解释，并证明了汇聚节点表示的有效性。

    Graph neural networks (GNNs) realize great success in graph learning but suffer from performance loss when meeting heterophily, i.e. neighboring nodes are dissimilar, due to their local and uniform aggregation. Existing attempts in incoorporating global aggregation for heterophilous GNNs usually require iteratively maintaining and updating full-graph information, which entails $\mathcal{O}(n^2)$ computation efficiency for a graph with $n$ nodes, leading to weak scalability to large graphs. In this paper, we propose SIMGA, a GNN structure integrating SimRank structural similarity measurement as global aggregation. The design of SIMGA is simple, yet it leads to promising results in both efficiency and effectiveness. The simplicity of SIMGA makes it the first heterophilous GNN model that can achieve a propagation efficiency near-linear to $n$. We theoretically demonstrate its effectiveness by treating SimRank as a new interpretation of GNN and prove that the aggregated node representation
    

