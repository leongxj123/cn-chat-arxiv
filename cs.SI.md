# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GFairHint: Improving Individual Fairness for Graph Neural Networks via Fairness Hint.](http://arxiv.org/abs/2305.15622) | GFairHint提出了一种新方法，通过辅助链接预测任务学习公平表示，并将其与原始图嵌入连接以增强图神经网络的个体公平性，同时不牺牲性能表现。 |

# 详细

[^1]: GFairHint：通过公平性提示提高图神经网络的个体公平性

    GFairHint: Improving Individual Fairness for Graph Neural Networks via Fairness Hint. (arXiv:2305.15622v1 [cs.LG])

    [http://arxiv.org/abs/2305.15622](http://arxiv.org/abs/2305.15622)

    GFairHint提出了一种新方法，通过辅助链接预测任务学习公平表示，并将其与原始图嵌入连接以增强图神经网络的个体公平性，同时不牺牲性能表现。

    

    鉴于机器学习中公平性问题日益受到关注以及图神经网络（GNN）在图数据学习上的卓越表现，GNN中的算法公平性受到了广泛关注。虽然许多现有的研究在群体层面上改善了公平性，但只有少数工作促进了个体公平性，这使得相似的个体具有相似的结果。促进个体公平性的理想框架应该（1）在公平性和性能之间平衡，（2）适应两种常用的个体相似性度量（从外部注释和从输入特征计算），（3）横跨各种GNN模型进行推广，（4）具有高效的计算能力。不幸的是，之前的工作都没有实现所有的理想条件。在这项工作中，我们提出了一种新的方法GFairHint，该方法通过辅助链接预测任务学习公平表示，然后将其与原始图嵌入连接以增强个体公平性。实验结果表明，GFairHint在不牺牲太多性能的情况下提高了个体公平性，并且优于目前的最新方法。

    Given the growing concerns about fairness in machine learning and the impressive performance of Graph Neural Networks (GNNs) on graph data learning, algorithmic fairness in GNNs has attracted significant attention. While many existing studies improve fairness at the group level, only a few works promote individual fairness, which renders similar outcomes for similar individuals. A desirable framework that promotes individual fairness should (1) balance between fairness and performance, (2) accommodate two commonly-used individual similarity measures (externally annotated and computed from input features), (3) generalize across various GNN models, and (4) be computationally efficient. Unfortunately, none of the prior work achieves all the desirables. In this work, we propose a novel method, GFairHint, which promotes individual fairness in GNNs and achieves all aforementioned desirables. GFairHint learns fairness representations through an auxiliary link prediction task, and then concate
    

