# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finding Decision Tree Splits in Streaming and Massively Parallel Models](https://arxiv.org/abs/2403.19867) | 提出了在数据流学习中计算决策树最佳分割点的算法，能够在流式计算和大规模并行模型中高效运行 |

# 详细

[^1]: 在流式和大规模并行模型中找到决策树分割点

    Finding Decision Tree Splits in Streaming and Massively Parallel Models

    [https://arxiv.org/abs/2403.19867](https://arxiv.org/abs/2403.19867)

    提出了在数据流学习中计算决策树最佳分割点的算法，能够在流式计算和大规模并行模型中高效运行

    

    在这项工作中，我们提出了一种数据流算法，用于计算决策树学习中的最优分割点。具体而言，给定观测数据流$x_i$及其标签$y_i$，目标是找到将数据分为两组的最佳分割点$j$，使得均方误差（回归问题）或误分类率（分类问题）最小化。我们提供了多种快速的数据流算法，这些算法在这些问题中使用亚线性空间和少量次数的遍历。这些算法还可以扩展到大规模并行计算模型中。尽管不能直接比较，但我们的工作与Domingos和Hulten的开创性工作（KDD 2000）相互补充。

    arXiv:2403.19867v1 Announce Type: cross  Abstract: In this work, we provide data stream algorithms that compute optimal splits in decision tree learning. In particular, given a data stream of observations $x_i$ and their labels $y_i$, the goal is to find the optimal split point $j$ that divides the data into two sets such that the mean squared error (for regression) or misclassification rate (for classification) is minimized. We provide various fast streaming algorithms that use sublinear space and a small number of passes for these problems. These algorithms can also be extended to the massively parallel computation model. Our work, while not directly comparable, complements the seminal work of Domingos and Hulten (KDD 2000).
    

