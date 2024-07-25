# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Unbiased Sparsification](https://arxiv.org/abs/2402.14925) | 该论文描述了对于排列不变或者可加可分的分裂函数，高效的无偏稀疏化特征。 |
| [^2] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |

# 详细

[^1]: 高效无偏稀疏化

    Efficient Unbiased Sparsification

    [https://arxiv.org/abs/2402.14925](https://arxiv.org/abs/2402.14925)

    该论文描述了对于排列不变或者可加可分的分裂函数，高效的无偏稀疏化特征。

    

    一个向量$p\in \mathbb{R}^n$的无偏$m$-稀疏化是一个具有平均值为$p$，最多有$m<n$个非零坐标的随机向量$Q\in \mathbb{R}^n。 无偏稀疏化可以压缩原始向量而不引入偏差；它出现在各种情境中，比如联邦学习和采样稀疏概率分布。 理想情况下，无偏稀疏化还应该最小化一个度量$Q$与原始$p$之间距离有多远的分裂函数$\mathsf{Div}(Q,p)$的期望值。 如果$Q$在这个意义上是最优的，那么我们称之为高效。 我们的主要结果描述了对于既是排列不变又是可加可分的分裂函数的高效无偏稀疏化。 令人惊讶的是，排列不变分裂函数的表征对于分裂函数的选择是健壮的，也就是说，我们针对平方欧氏距离的最优$Q$的类与我们的类重合了op

    arXiv:2402.14925v1 Announce Type: cross  Abstract: An unbiased $m$-sparsification of a vector $p\in \mathbb{R}^n$ is a random vector $Q\in \mathbb{R}^n$ with mean $p$ that has at most $m<n$ nonzero coordinates. Unbiased sparsification compresses the original vector without introducing bias; it arises in various contexts, such as in federated learning and sampling sparse probability distributions. Ideally, unbiased sparsification should also minimize the expected value of a divergence function $\mathsf{Div}(Q,p)$ that measures how far away $Q$ is from the original $p$. If $Q$ is optimal in this sense, then we call it efficient. Our main results describe efficient unbiased sparsifications for divergences that are either permutation-invariant or additively separable. Surprisingly, the characterization for permutation-invariant divergences is robust to the choice of divergence function, in the sense that our class of optimal $Q$ for squared Euclidean distance coincides with our class of op
    
[^2]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    

