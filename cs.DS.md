# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments.](http://arxiv.org/abs/2401.13185) | 本文提出了三种高效计算训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$的算法，相比于以前的工作，这些算法能够显著加速交叉验证，而无需重新计算矩阵乘积或统计量。 |

# 详细

[^1]: 简化交叉验证：高效地计算不需要全量重新计算矩阵乘积或统计量的列向中心化和标准化训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$

    Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments. (arXiv:2401.13185v1 [cs.LG])

    [http://arxiv.org/abs/2401.13185](http://arxiv.org/abs/2401.13185)

    本文提出了三种高效计算训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$的算法，相比于以前的工作，这些算法能够显著加速交叉验证，而无需重新计算矩阵乘积或统计量。

    

    交叉验证是一种广泛使用的评估预测模型在未知数据上表现的技术。许多预测模型，如基于核的偏最小二乘（PLS）模型，需要仅使用输入矩阵$\mathbf{X}$和输出矩阵$\mathbf{Y}$中的训练集样本来计算$\mathbf{X}^{\mathbf{T}}\mathbf{X}$和$\mathbf{X}^{\mathbf{T}}\mathbf{Y}$。在这项工作中，我们提出了三种高效计算这些矩阵的算法。第一种算法不需要列向预处理。第二种算法允许以训练集均值为中心化点进行列向中心化。第三种算法允许以训练集均值和标准差为中心化点和标准化点进行列向中心化和标准化。通过证明正确性和优越的计算复杂度，它们相比于直接交叉验证和以前的快速交叉验证工作，提供了显著的交叉验证加速，而无需数据泄露。它们适合并行计算。

    Cross-validation is a widely used technique for assessing the performance of predictive models on unseen data. Many predictive models, such as Kernel-Based Partial Least-Squares (PLS) models, require the computation of $\mathbf{X}^{\mathbf{T}}\mathbf{X}$ and $\mathbf{X}^{\mathbf{T}}\mathbf{Y}$ using only training set samples from the input and output matrices, $\mathbf{X}$ and $\mathbf{Y}$, respectively. In this work, we present three algorithms that efficiently compute these matrices. The first one allows no column-wise preprocessing. The second one allows column-wise centering around the training set means. The third one allows column-wise centering and column-wise scaling around the training set means and standard deviations. Demonstrating correctness and superior computational complexity, they offer significant cross-validation speedup compared with straight-forward cross-validation and previous work on fast cross-validation - all without data leakage. Their suitability for paralle
    

