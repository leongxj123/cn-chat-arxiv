# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse PCA With Multiple Components.](http://arxiv.org/abs/2209.14790) | 本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。 |

# 详细

[^1]: 多组分的稀疏主成分分析

    Sparse PCA With Multiple Components. (arXiv:2209.14790v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.14790](http://arxiv.org/abs/2209.14790)

    本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。

    

    稀疏主成分分析是一种用于以可解释的方式解释高维数据集方差的基本技术。这涉及解决一个稀疏性和正交性约束的凸最大化问题，其计算复杂度非常高。大多数现有的方法通过迭代计算一个稀疏主成分并缩减协方差矩阵来解决稀疏主成分分析，但在寻找多个相互正交的主成分时，这些方法不能保证所得解的正交性和最优性。我们挑战这种现状，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。此外，我们采用另一种方法来加强上界，我们使用额外的二阶锥不等式来加强上界。

    Sparse Principal Component Analysis (sPCA) is a cardinal technique for obtaining combinations of features, or principal components (PCs), that explain the variance of high-dimensional datasets in an interpretable manner. This involves solving a sparsity and orthogonality constrained convex maximization problem, which is extremely computationally challenging. Most existing works address sparse PCA via methods-such as iteratively computing one sparse PC and deflating the covariance matrix-that do not guarantee the orthogonality, let alone the optimality, of the resulting solution when we seek multiple mutually orthogonal PCs. We challenge this status by reformulating the orthogonality conditions as rank constraints and optimizing over the sparsity and rank constraints simultaneously. We design tight semidefinite relaxations to supply high-quality upper bounds, which we strengthen via additional second-order cone inequalities when each PC's individual sparsity is specified. Further, we de
    

