# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OpenGraph: Towards Open Graph Foundation Models](https://arxiv.org/abs/2403.01121) | 该论文旨在通过开发一个通用图基础模型，以解决现有图神经网络在泛化到与训练数据显著不同的未见图数据时遇到的困难。 |
| [^2] | [Clarify Confused Nodes Through Separated Learning.](http://arxiv.org/abs/2306.02285) | 本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。 |

# 详细

[^1]: OpenGraph: 迈向开放图基础模型

    OpenGraph: Towards Open Graph Foundation Models

    [https://arxiv.org/abs/2403.01121](https://arxiv.org/abs/2403.01121)

    该论文旨在通过开发一个通用图基础模型，以解决现有图神经网络在泛化到与训练数据显著不同的未见图数据时遇到的困难。

    

    arXiv:2403.01121v1 公告类型: 跨交互   摘要: 图学习已成为解释和利用各领域的关系数据的不可或缺部分，从推荐系统到社交网络分析。在这种背景下，各种GNN已经成为编码图的结构信息的有希望的方法论，通过有效地捕捉图的潜在结构，这些GNN已经展示出在增强图学习任务性能方面的巨大潜力，例如链接预测和节点分类。然而，尽管取得了成功，一个显著的挑战仍然存在: 这些先进方法通常在将显著不同于训练实例的未见图数据泛化时遇到困难。在这项工作中，我们的目标是通过开发一个通用图基础模型来推进图学习范式。该模型旨在理解多样图数据中存在的复杂拓扑模式，使其在零-shot情况下表现出色。

    arXiv:2403.01121v1 Announce Type: cross  Abstract: Graph learning has become indispensable for interpreting and harnessing relational data in diverse fields, ranging from recommendation systems to social network analysis. In this context, a variety of GNNs have emerged as promising methodologies for encoding the structural information of graphs. By effectively capturing the graph's underlying structure, these GNNs have shown great potential in enhancing performance in graph learning tasks, such as link prediction and node classification. However, despite their successes, a significant challenge persists: these advanced methods often face difficulties in generalizing to unseen graph data that significantly differs from the training instances. In this work, our aim is to advance the graph learning paradigm by developing a general graph foundation model. This model is designed to understand the complex topological patterns present in diverse graph data, enabling it to excel in zero-shot g
    
[^2]: 通过分离学习解决混淆节点问题

    Clarify Confused Nodes Through Separated Learning. (arXiv:2306.02285v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02285](http://arxiv.org/abs/2306.02285)

    本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。

    

    图神经网络（GNN）在图导向任务中取得了显著的进展。然而，现实世界的图中不可避免地包含一定比例的异质节点，这挑战了经典GNN的同质性假设，并阻碍了其性能。现有研究大多数仍设计了具有异质节点和同质节点间共享权重的通用模型。尽管这些努力中包含了高阶信息和多通道架构，但往往效果不佳。少数研究尝试训练不同节点组的分离学习，但受到了不合适的分离度量和低效率的影响。本文首先提出了一种新的度量指标，称为邻域混淆（NC），以便更可靠地分离节点。我们观察到具有不同NC值的节点组在组内准确度和可视化嵌入上存在一定差异。这为基于邻域混淆的图卷积网络（NC-GCN）铺平了道路。

    Graph neural networks (GNNs) have achieved remarkable advances in graph-oriented tasks. However, real-world graphs invariably contain a certain proportion of heterophilous nodes, challenging the homophily assumption of classical GNNs and hindering their performance. Most existing studies continue to design generic models with shared weights between heterophilous and homophilous nodes. Despite the incorporation of high-order messages or multi-channel architectures, these efforts often fall short. A minority of studies attempt to train different node groups separately but suffer from inappropriate separation metrics and low efficiency. In this paper, we first propose a new metric, termed Neighborhood Confusion (NC), to facilitate a more reliable separation of nodes. We observe that node groups with different levels of NC values exhibit certain differences in intra-group accuracy and visualized embeddings. These pave the way for Neighborhood Confusion-guided Graph Convolutional Network (N
    

