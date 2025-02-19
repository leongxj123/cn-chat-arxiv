# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast and Efficient Matching Algorithm with Deadline Instances](https://arxiv.org/abs/2305.08353) | 本文介绍了一种带有截止期限实例的快速高效匹配算法，通过引入带有截止期限的市场模型，提出了两种优化算法（FastGreedy和FastPostponedGreedy）。该算法在处理机器学习中的在线加权匹配问题时具有较快的速度和准确性。 |

# 详细

[^1]: 快速高效的带有截止期限实例的匹配算法

    Fast and Efficient Matching Algorithm with Deadline Instances

    [https://arxiv.org/abs/2305.08353](https://arxiv.org/abs/2305.08353)

    本文介绍了一种带有截止期限实例的快速高效匹配算法，通过引入带有截止期限的市场模型，提出了两种优化算法（FastGreedy和FastPostponedGreedy）。该算法在处理机器学习中的在线加权匹配问题时具有较快的速度和准确性。

    

    在机器学习中，在线加权匹配问题由于其众多应用而成为一个基本问题。尽管在这个领域已经做了很多努力，但现有的算法要么速度太慢，要么没有考虑到截止期限（节点可以匹配的最长时间）。在本文中，我们首先引入了一个带有截止期限的市场模型。接下来，我们提出了两个优化算法（FastGreedy和FastPostponedGreedy），并给出了关于算法时间复杂度和正确性的理论证明。在FastGreedy算法中，我们已经知道一个节点是买家还是卖家。但在FastPostponedGreedy算法中，一开始我们不知道每个节点的状态。然后，我们推广了一个草图矩阵，以在真实数据集和合成数据集上运行原始算法和我们的算法。设 ε ∈（0,0.1）表示每条边的真实权重的相对误差。原始的Greedy和Po算法的竞争比率是多少。

    The online weighted matching problem is a fundamental problem in machine learning due to its numerous applications. Despite many efforts in this area, existing algorithms are either too slow or don't take $\mathrm{deadline}$ (the longest time a node can be matched) into account. In this paper, we introduce a market model with $\mathrm{deadline}$ first. Next, we present our two optimized algorithms (\textsc{FastGreedy} and \textsc{FastPostponedGreedy}) and offer theoretical proof of the time complexity and correctness of our algorithms. In \textsc{FastGreedy} algorithm, we have already known if a node is a buyer or a seller. But in \textsc{FastPostponedGreedy} algorithm, the status of each node is unknown at first. Then, we generalize a sketching matrix to run the original and our algorithms on both real data sets and synthetic data sets. Let $\epsilon \in (0,0.1)$ denote the relative error of the real weight of each edge. The competitive ratio of original \textsc{Greedy} and \textsc{Po
    

