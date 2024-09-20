# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The complementary contributions of academia and industry to AI research.](http://arxiv.org/abs/2401.10268) | 工业界的研究团队在人工智能研究中有更高的关注度和引用率，且更有可能产生最先进的模型。而学术界的团队则更倾向于产生具有更高程度创新的工作，出现非常不寻常和典型的论文。这种影响力-创新度优势在不同领域、团队规模、资历和声望下均存在。 |
| [^2] | [Self-attention Dual Embedding for Graphs with Heterophily.](http://arxiv.org/abs/2305.18385) | 本研究提出了一种新颖的图神经网络，采用自注意力机制，适用于异质性图和同质性图，并在许多标准数据集上展示出最先进的性能。 |
| [^3] | [Node Feature Augmentation Vitaminizes Network Alignment.](http://arxiv.org/abs/2304.12751) | 本研究提出了Grad-Align+方法，通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法，解决了NA方法缺乏额外信息的问题。 |

# 详细

[^1]: 学术界和工业界对人工智能研究的互补贡献

    The complementary contributions of academia and industry to AI research. (arXiv:2401.10268v1 [cs.CY])

    [http://arxiv.org/abs/2401.10268](http://arxiv.org/abs/2401.10268)

    工业界的研究团队在人工智能研究中有更高的关注度和引用率，且更有可能产生最先进的模型。而学术界的团队则更倾向于产生具有更高程度创新的工作，出现非常不寻常和典型的论文。这种影响力-创新度优势在不同领域、团队规模、资历和声望下均存在。

    

    人工智能在工业界和学术界中都取得了巨大的发展。然而，工业界近期的突破性进展引起了人们对学术研究在该领域中的作用的新视角。在这里，我们对过去25年中两个环境中产生的人工智能的影响和类型进行了描述，并建立了几种模式。我们发现，由工业界研究人员组成的团队发表的文章往往更受关注，具有更高的被引用和引发引用颠覆的可能性，且更有可能产生最先进的模型。相反，我们发现纯学术团队发表了大部分的人工智能研究，并倾向于产生更高程度的创新工作，单篇论文有数倍的可能性是非常不寻常和典型的。工业界和学术界在影响力-创新度方面的优势不受子领域、团队规模、资历和声望的影响。我们发现学术界产生了更多综述和分析型论文，而工业界则更多地注重应用和技术发展。

    Artificial intelligence (AI) has seen tremendous development in industry and academia. However, striking recent advances by industry have stunned the world, inviting a fresh perspective on the role of academic research in this field. Here, we characterize the impact and type of AI produced by both environments over the last 25 years and establish several patterns. We find that articles published by teams consisting exclusively of industry researchers tend to get greater attention, with a higher chance of being highly cited and citation-disruptive, and several times more likely to produce state-of-the-art models. In contrast, we find that exclusively academic teams publish the bulk of AI research and tend to produce higher novelty work, with single papers having several times higher likelihood of being unconventional and atypical. The respective impact-novelty advantages of industry and academia are robust to controls for subfield, team size, seniority, and prestige. We find that academ
    
[^2]: 自注意力双重嵌入：适用于异质性图的图神经网络

    Self-attention Dual Embedding for Graphs with Heterophily. (arXiv:2305.18385v1 [cs.LG])

    [http://arxiv.org/abs/2305.18385](http://arxiv.org/abs/2305.18385)

    本研究提出了一种新颖的图神经网络，采用自注意力机制，适用于异质性图和同质性图，并在许多标准数据集上展示出最先进的性能。

    

    图神经网络（GNNs）在节点分类任务中取得了重大成功。GNNs通常假设图是同质的，即相邻节点很可能属于相同的类别。然而，许多真实世界的图都是异质的，这导致使用标准的GNNs时分类精度要低得多。在本文中，我们设计了一种新颖的GNN，它对异质性和同质性图都有效。我们的工作基于三个主要观察结果。首先，我们展示了在不同的图中，节点特征和图拓扑提供不同数量的信息，因此应该独立编码并以自适应方式优先级化。其次，我们展示了当传播图拓扑信息时允许负的注意权重可以提高精度。最后，我们展示了节点之间不对称的注意权重是有帮助的。我们设计了一种GNN，利用这些观察结果通过新颖的自注意力机制。我们评估了我们的算法在一些标准的节点分类数据集上，并展示了在同质性和异质性图上的最新性能。

    Graph Neural Networks (GNNs) have been highly successful for the node classification task. GNNs typically assume graphs are homophilic, i.e. neighboring nodes are likely to belong to the same class. However, a number of real-world graphs are heterophilic, and this leads to much lower classification accuracy using standard GNNs. In this work, we design a novel GNN which is effective for both heterophilic and homophilic graphs. Our work is based on three main observations. First, we show that node features and graph topology provide different amounts of informativeness in different graphs, and therefore they should be encoded independently and prioritized in an adaptive manner. Second, we show that allowing negative attention weights when propagating graph topology information improves accuracy. Finally, we show that asymmetric attention weights between nodes are helpful. We design a GNN which makes use of these observations through a novel self-attention mechanism. We evaluate our algor
    
[^3]: 节点特征增强改进网络对齐

    Node Feature Augmentation Vitaminizes Network Alignment. (arXiv:2304.12751v1 [cs.SI])

    [http://arxiv.org/abs/2304.12751](http://arxiv.org/abs/2304.12751)

    本研究提出了Grad-Align+方法，通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法，解决了NA方法缺乏额外信息的问题。

    

    网络对齐（NA）是通过给定网络的拓扑和/或特征信息来发现多个网络之间的节点对应关系的任务。虽然NA方法在各种场景下取得了显著的成功，但其有效性并不总是有额外信息，如先前的锚点链接和/或节点特征。为了解决这个实际的挑战，我们提出了Grad-Align+，这是一种新颖的NA方法，建立在最近一种最先进的NA方法Grad-Align之上，Grad-Align+仅逐步发现部分节点对，直到找到所有节点对。在设计Grad-Align+时，我们考虑如何通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法。为了实现这个目标，我们开发了由三个关键组成部分组成的Grad-Align+：基于中心性的节点特征增强（CNFA）、图切片生成和优化节点嵌入特征（ONIFE）。

    Network alignment (NA) is the task of discovering node correspondences across multiple networks using topological and/or feature information of given networks. Although NA methods have achieved remarkable success in a myriad of scenarios, their effectiveness is not without additional information such as prior anchor links and/or node features, which may not always be available due to privacy concerns or access restrictions. To tackle this practical challenge, we propose Grad-Align+, a novel NA method built upon a recent state-of-the-art NA method, the so-called Grad-Align, that gradually discovers only a part of node pairs until all node pairs are found. In designing Grad-Align+, we account for how to augment node features in the sense of performing the NA task and how to design our NA method by maximally exploiting the augmented node features. To achieve this goal, we develop Grad-Align+ consisting of three key components: 1) centrality-based node feature augmentation (CNFA), 2) graph
    

