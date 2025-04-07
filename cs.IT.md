# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Orthogonal Non-negative Matrix Factorization: a Maximum-Entropy-Principle Approach.](http://arxiv.org/abs/2210.02672) | 本文提出了一种新的解决正交非负矩阵分解问题的方法，该方法使用了基于最大熵原则的解决方案，并保证了矩阵的正交性和稀疏性以及非负性。该方法在不影响近似质量的情况下具有较好的性能速度和优于文献中类似方法的稀疏性、正交性。 |

# 详细

[^1]: 正交非负矩阵分解:最大熵原则方法

    Orthogonal Non-negative Matrix Factorization: a Maximum-Entropy-Principle Approach. (arXiv:2210.02672v2 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2210.02672](http://arxiv.org/abs/2210.02672)

    本文提出了一种新的解决正交非负矩阵分解问题的方法，该方法使用了基于最大熵原则的解决方案，并保证了矩阵的正交性和稀疏性以及非负性。该方法在不影响近似质量的情况下具有较好的性能速度和优于文献中类似方法的稀疏性、正交性。

    

    本文提出了一种解决正交非负矩阵分解（ONMF）问题的新方法，该问题的目标是通过两个非负矩阵（特征矩阵和混合矩阵）的乘积来近似输入数据矩阵，其中一个矩阵是正交的。我们展示了如何将ONMF解释为特定的设施定位问题，并针对ONMF问题采用基于最大熵原则的FLP解决方案进行了调整。所提出的方法保证了特征矩阵或混合矩阵的正交性和稀疏性，同时确保了两者的非负性。此外，我们的方法还开发了一个定量的“真实”潜在特征数量的特征-超参数用于ONMF。针对合成数据集以及标准的基因芯片数组数据集进行的评估表明，该方法在不影响近似质量的情况下具有较好的稀疏性、正交性和性能速度，相对于文献中类似方法有显著的改善。

    In this paper, we introduce a new methodology to solve the orthogonal nonnegative matrix factorization (ONMF) problem, where the objective is to approximate an input data matrix by a product of two nonnegative matrices, the features matrix and the mixing matrix, where one of them is orthogonal. We show how the ONMF can be interpreted as a specific facility-location problem (FLP), and adapt a maximum-entropy-principle based solution for FLP to the ONMF problem. The proposed approach guarantees orthogonality and sparsity of the features or the mixing matrix, while ensuring nonnegativity of both. Additionally, our methodology develops a quantitative characterization of ``true" number of underlying features - a hyperparameter required for the ONMF. An evaluation of the proposed method conducted on synthetic datasets, as well as a standard genetic microarray dataset indicates significantly better sparsity, orthogonality, and performance speed compared to similar methods in the literature, w
    

