# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Node Feature Augmentation Vitaminizes Network Alignment.](http://arxiv.org/abs/2304.12751) | 本研究提出了Grad-Align+方法，通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法，解决了NA方法缺乏额外信息的问题。 |

# 详细

[^1]: 节点特征增强改进网络对齐

    Node Feature Augmentation Vitaminizes Network Alignment. (arXiv:2304.12751v1 [cs.SI])

    [http://arxiv.org/abs/2304.12751](http://arxiv.org/abs/2304.12751)

    本研究提出了Grad-Align+方法，通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法，解决了NA方法缺乏额外信息的问题。

    

    网络对齐（NA）是通过给定网络的拓扑和/或特征信息来发现多个网络之间的节点对应关系的任务。虽然NA方法在各种场景下取得了显著的成功，但其有效性并不总是有额外信息，如先前的锚点链接和/或节点特征。为了解决这个实际的挑战，我们提出了Grad-Align+，这是一种新颖的NA方法，建立在最近一种最先进的NA方法Grad-Align之上，Grad-Align+仅逐步发现部分节点对，直到找到所有节点对。在设计Grad-Align+时，我们考虑如何通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法。为了实现这个目标，我们开发了由三个关键组成部分组成的Grad-Align+：基于中心性的节点特征增强（CNFA）、图切片生成和优化节点嵌入特征（ONIFE）。

    Network alignment (NA) is the task of discovering node correspondences across multiple networks using topological and/or feature information of given networks. Although NA methods have achieved remarkable success in a myriad of scenarios, their effectiveness is not without additional information such as prior anchor links and/or node features, which may not always be available due to privacy concerns or access restrictions. To tackle this practical challenge, we propose Grad-Align+, a novel NA method built upon a recent state-of-the-art NA method, the so-called Grad-Align, that gradually discovers only a part of node pairs until all node pairs are found. In designing Grad-Align+, we account for how to augment node features in the sense of performing the NA task and how to design our NA method by maximally exploiting the augmented node features. To achieve this goal, we develop Grad-Align+ consisting of three key components: 1) centrality-based node feature augmentation (CNFA), 2) graph
    

