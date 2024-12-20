# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Strong consistency and optimality of spectral clustering in symmetric binary non-uniform Hypergraph Stochastic Block Model.](http://arxiv.org/abs/2306.06845) | 论文提出了非均匀超图随机块模型下谱聚类的强一致性信息理论阈值，并且在该阈值以下给出估计标签的期望“不匹配率”上界。并且，单步谱算法可以在超过该阈值时非常高的概率正确地给定每个顶点的标签。 |

# 详细

[^1]: 对称二元非均匀超图随机块模型中谱聚类的强一致性与最优性

    Strong consistency and optimality of spectral clustering in symmetric binary non-uniform Hypergraph Stochastic Block Model. (arXiv:2306.06845v1 [math.ST])

    [http://arxiv.org/abs/2306.06845](http://arxiv.org/abs/2306.06845)

    论文提出了非均匀超图随机块模型下谱聚类的强一致性信息理论阈值，并且在该阈值以下给出估计标签的期望“不匹配率”上界。并且，单步谱算法可以在超过该阈值时非常高的概率正确地给定每个顶点的标签。

    

    本论文考虑了在非均匀超图随机块模型下，两个等大小的社区（n/2）中的随机超图上的无监督分类问题，其中每个边只依赖于其顶点的标签，边以一定概率独立出现。在这篇论文中，建立了强一致性的信息理论阈值，在该阈值以下，任何算法都有很高概率会误分类至少两个顶点，而特征向量估计量的期望“不匹配率”上界为$n$的阈值的负指数。另一方面，当超过该阈值时，尽管张量收缩引起了信息损失，但单步谱算法仅在给定收缩的邻接矩阵时，即使SDP在某些情况下失败，也可以非常高的概率正确地给定每个顶点分配标签。此外，强一致性可以通过对所有次优聚合信息实现。

    Consider the unsupervised classification problem in random hypergraphs under the non-uniform \emph{Hypergraph Stochastic Block Model} (HSBM) with two equal-sized communities ($n/2$), where each edge appears independently with some probability depending only on the labels of its vertices. In this paper, an \emph{information-theoretical} threshold for strong consistency is established. Below the threshold, every algorithm would misclassify at least two vertices with high probability, and the expected \emph{mismatch ratio} of the eigenvector estimator is upper bounded by $n$ to the power of minus the threshold. On the other hand, when above the threshold, despite the information loss induced by tensor contraction, one-stage spectral algorithms assign every vertex correctly with high probability when only given the contracted adjacency matrix, even if \emph{semidefinite programming} (SDP) fails in some scenarios. Moreover, strong consistency is achievable by aggregating information from al
    

