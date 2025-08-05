# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Node Duplication Improves Cold-start Link Prediction](https://arxiv.org/abs/2402.09711) | 本文研究了在链路预测中改进GNN在低度节点上的性能，提出了一种名为NodeDup的增强技术，通过复制低度节点并创建链接来提高性能。 |

# 详细

[^1]: 节点复制改善冷启动链路预测

    Node Duplication Improves Cold-start Link Prediction

    [https://arxiv.org/abs/2402.09711](https://arxiv.org/abs/2402.09711)

    本文研究了在链路预测中改进GNN在低度节点上的性能，提出了一种名为NodeDup的增强技术，通过复制低度节点并创建链接来提高性能。

    

    图神经网络（GNN）在图机器学习中非常突出，并在链路预测（LP）任务中展现了最先进的性能。然而，最近的研究表明，尽管整体上表现出色，GNN在低度节点上的表现却较差。在推荐系统等LP的实际应用中，改善低度节点的性能至关重要，因为这等同于解决冷启动问题，提高用户在少数观察的相互作用中的体验。本文研究了改进GNN在低度节点上的LP性能，同时保持其在高度节点上的性能，并提出了一种简单但非常有效的增强技术，称为NodeDup。具体而言，NodeDup在标准的监督LP训练方案中，在低度节点上复制节点并在节点和其副本之间创建链接。通过利用“多视图”视角，该方法可以显著提高LP的性能。

    arXiv:2402.09711v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) are prominent in graph machine learning and have shown state-of-the-art performance in Link Prediction (LP) tasks. Nonetheless, recent studies show that GNNs struggle to produce good results on low-degree nodes despite their overall strong performance. In practical applications of LP, like recommendation systems, improving performance on low-degree nodes is critical, as it amounts to tackling the cold-start problem of improving the experiences of users with few observed interactions. In this paper, we investigate improving GNNs' LP performance on low-degree nodes while preserving their performance on high-degree nodes and propose a simple yet surprisingly effective augmentation technique called NodeDup. Specifically, NodeDup duplicates low-degree nodes and creates links between nodes and their own duplicates before following the standard supervised LP training scheme. By leveraging a ''multi-view'' perspectiv
    

