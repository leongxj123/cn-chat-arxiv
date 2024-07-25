# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |

# 详细

[^1]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    

