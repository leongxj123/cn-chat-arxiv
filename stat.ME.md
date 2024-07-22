# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithms for Non-Negative Matrix Factorization on Noisy Data With Negative Values](https://arxiv.org/abs/2311.04855) | 本文提出了两种算法，Shift-NMF和Nearly-NMF，可以处理带有负值的嘈杂数据，在不引入正的偏移量的情况下正确恢复非负信号。 |

# 详细

[^1]: 用于处理带有负值的嘈杂数据的非负矩阵分解算法

    Algorithms for Non-Negative Matrix Factorization on Noisy Data With Negative Values

    [https://arxiv.org/abs/2311.04855](https://arxiv.org/abs/2311.04855)

    本文提出了两种算法，Shift-NMF和Nearly-NMF，可以处理带有负值的嘈杂数据，在不引入正的偏移量的情况下正确恢复非负信号。

    

    非负矩阵分解（NMF）是一种降维技术，在分析嘈杂数据，特别是天文数据方面表现出了潜力。在这些数据集中，由于噪声，观测数据可能包含负值，即使真实的物理信号严格为正。以往的NMF工作未以统计一致的方式处理负数据，这在低信噪比数据中出现许多负值时会变得棘手。在本文中，我们提出了两种算法，Shift-NMF和Nearly-NMF，可以处理输入数据的嘈杂性，并消除任何引入的负值。这两种算法都使用负数据空间而不进行截取，并且在消除负数据时不会引入正的偏移量。我们在简单和更现实的示例上进行了数值演示，并证明了这两种算法具有单调性。

    arXiv:2311.04855v2 Announce Type: replace-cross  Abstract: Non-negative matrix factorization (NMF) is a dimensionality reduction technique that has shown promise for analyzing noisy data, especially astronomical data. For these datasets, the observed data may contain negative values due to noise even when the true underlying physical signal is strictly positive. Prior NMF work has not treated negative data in a statistically consistent manner, which becomes problematic for low signal-to-noise data with many negative values. In this paper we present two algorithms, Shift-NMF and Nearly-NMF, that can handle both the noisiness of the input data and also any introduced negativity. Both of these algorithms use the negative data space without clipping, and correctly recover non-negative signals without any introduced positive offset that occurs when clipping negative data. We demonstrate this numerically on both simple and more realistic examples, and prove that both algorithms have monotoni
    

