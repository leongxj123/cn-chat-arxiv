# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Low-Rank Graph Contrastive Learning for Node Classification](https://arxiv.org/abs/2402.09600) | 本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。 |

# 详细

[^1]: 低秩图对比学习用于节点分类

    Low-Rank Graph Contrastive Learning for Node Classification

    [https://arxiv.org/abs/2402.09600](https://arxiv.org/abs/2402.09600)

    本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。

    

    图神经网络（GNNs）广泛应用于学习节点表示，并在节点分类等各种任务中表现出色。然而，最近的研究表明，在现实世界的图数据中不可避免地存在噪声，这会严重降低GNNs的性能。在本文中，我们提出了一种新颖且鲁棒的GNN编码器，即低秩图对比学习（LR-GCL）。我们的方法通过两个步骤进行转导节点分类。首先，通过低秩正常对比学习训练一个名为LR-GCL的低秩GCL编码器。然后，使用LR-GCL生成的特征，使用线性转导分类算法对图中的未标记节点进行分类。我们的LR-GCL受到图数据和其标签的低频性质的启示，并在理论上受到我们关于转导学习的尖锐泛化界限的推动。

    arXiv:2402.09600v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have been widely used to learn node representations and with outstanding performance on various tasks such as node classification. However, noise, which inevitably exists in real-world graph data, would considerably degrade the performance of GNNs revealed by recent studies. In this work, we propose a novel and robust GNN encoder, Low-Rank Graph Contrastive Learning (LR-GCL). Our method performs transductive node classification in two steps. First, a low-rank GCL encoder named LR-GCL is trained by prototypical contrastive learning with low-rank regularization. Next, using the features produced by LR-GCL, a linear transductive classification algorithm is used to classify the unlabeled nodes in the graph. Our LR-GCL is inspired by the low frequency property of the graph data and its labels, and it is also theoretically motivated by our sharp generalization bound for transductive learning. To the best of our kno
    

