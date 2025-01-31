# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Hyperedge Prediction with Context-Aware Self-Supervised Learning.](http://arxiv.org/abs/2309.05798) | 该论文提出了一种增强超边预测的方法，通过上下文感知的节点聚合和自监督对比学习来解决超边预测中的问题。这种方法可以准确捕捉节点之间的复杂关系，并缓解数据稀疏问题。 |

# 详细

[^1]: 增强上下文感知自监督学习的超边预测

    Enhancing Hyperedge Prediction with Context-Aware Self-Supervised Learning. (arXiv:2309.05798v1 [cs.LG])

    [http://arxiv.org/abs/2309.05798](http://arxiv.org/abs/2309.05798)

    该论文提出了一种增强超边预测的方法，通过上下文感知的节点聚合和自监督对比学习来解决超边预测中的问题。这种方法可以准确捕捉节点之间的复杂关系，并缓解数据稀疏问题。

    

    超图可以自然地建模群组关系（例如，一组共同购买物品的用户），hyperedge预测是预测未来或未观察到的超边的任务，在许多实际应用中都非常重要。然而，目前的研究中很少探讨以下挑战：（C1）如何聚合每个超边候选中的节点以准确预测超边？（C2）如何缓解超边预测中固有的数据稀疏问题？为了同时解决这两个挑战，本文提出了一种新颖的超边预测框架CASH，它采用了（1）上下文感知节点聚合，精确捕捉每个超边中节点之间的复杂关系，用于解决挑战（C1），以及（2）自监督对比学习在超边预测上下文中增强超图表示，以应对挑战（C2）。此外，针对挑战（C2），我们提出了超边感知的数据增强方法。

    Hypergraphs can naturally model group-wise relations (e.g., a group of users who co-purchase an item) as hyperedges. Hyperedge prediction is to predict future or unobserved hyperedges, which is a fundamental task in many real-world applications (e.g., group recommendation). Despite the recent breakthrough of hyperedge prediction methods, the following challenges have been rarely studied: (C1) How to aggregate the nodes in each hyperedge candidate for accurate hyperedge prediction? and (C2) How to mitigate the inherent data sparsity problem in hyperedge prediction? To tackle both challenges together, in this paper, we propose a novel hyperedge prediction framework (CASH) that employs (1) context-aware node aggregation to precisely capture complex relations among nodes in each hyperedge for (C1) and (2) self-supervised contrastive learning in the context of hyperedge prediction to enhance hypergraph representations for (C2). Furthermore, as for (C2), we propose a hyperedge-aware augmenta
    

