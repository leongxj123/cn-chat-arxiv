# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Randomized Algorithms for Symmetric Nonnegative Matrix Factorization](https://arxiv.org/abs/2402.08134) | 本论文提出了两种随机算法来更快、更可扩展地计算对称非负矩阵分解，其中一种使用随机矩阵草图来计算初始低秩输入矩阵，另一种使用随机杠杆得分采样来近似解决约束最小二乘问题。这些方法在大规模真实世界的图聚类任务上取得了良好的效果。 |

# 详细

[^1]: 对称非负矩阵分解的随机算法

    Randomized Algorithms for Symmetric Nonnegative Matrix Factorization

    [https://arxiv.org/abs/2402.08134](https://arxiv.org/abs/2402.08134)

    本论文提出了两种随机算法来更快、更可扩展地计算对称非负矩阵分解，其中一种使用随机矩阵草图来计算初始低秩输入矩阵，另一种使用随机杠杆得分采样来近似解决约束最小二乘问题。这些方法在大规模真实世界的图聚类任务上取得了良好的效果。

    

    对称非负矩阵分解（SymNMF）是数据分析和机器学习中一种将对称矩阵近似表示为非负、低秩矩阵及其转置的技术。为了设计更快、更可扩展的SymNMF算法，我们开发了两种随机算法来进行计算。第一种算法使用随机矩阵草图计算初始的低秩输入矩阵，并利用该输入迅速计算SymNMF。第二种算法使用随机杠杆得分采样来近似解决约束最小二乘问题。许多成功的SymNMF方法依赖于（近似）解决一系列约束最小二乘问题。我们在理论上证明了杠杆得分采样可以以高概率近似解决非负最小二乘问题，达到所选精度。最后，我们通过将这两种方法应用于大规模真实世界的图聚类任务中，证明了它们的有效性。

    Symmetric Nonnegative Matrix Factorization (SymNMF) is a technique in data analysis and machine learning that approximates a symmetric matrix with a product of a nonnegative, low-rank matrix and its transpose. To design faster and more scalable algorithms for SymNMF we develop two randomized algorithms for its computation. The first algorithm uses randomized matrix sketching to compute an initial low-rank input matrix and proceeds to use this input to rapidly compute a SymNMF. The second algorithm uses randomized leverage score sampling to approximately solve constrained least squares problems. Many successful methods for SymNMF rely on (approximately) solving sequences of constrained least squares problems. We prove theoretically that leverage score sampling can approximately solve nonnegative least squares problems to a chosen accuracy with high probability. Finally we demonstrate that both methods work well in practice by applying them to graph clustering tasks on large real world d
    

