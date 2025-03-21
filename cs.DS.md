# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect.](http://arxiv.org/abs/2401.00422) | 这篇论文从理论和实证分析的角度深入研究了维度诅咒的两个主要原因——距离集中和流形效应，并通过实验证明了使用Minkowski距离进行最近邻搜索（NNS）在高维数据中取得了最佳性能。 |

# 详细

[^1]: 从距离集中和流形效应解读维度诅咒

    Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect. (arXiv:2401.00422v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00422](http://arxiv.org/abs/2401.00422)

    这篇论文从理论和实证分析的角度深入研究了维度诅咒的两个主要原因——距离集中和流形效应，并通过实验证明了使用Minkowski距离进行最近邻搜索（NNS）在高维数据中取得了最佳性能。

    

    随着维度的增加，数据的特征如分布和异质性变得越来越复杂和违反直觉。这种现象被称为维度诅咒，低维空间中成立的常见模式和关系（例如内部和边界模式）在高维空间中可能无效。这导致回归、分类或聚类模型或算法的性能降低。维度诅咒可以归因于许多原因。本文首先总结了与处理高维数据相关的五个挑战，并解释了回归、分类或聚类任务失败的潜在原因。随后，我们通过理论和实证分析深入研究了维度诅咒的两个主要原因，即距离集中和流形效应。结果表明，使用三种典型的距离测量进行最近邻搜索（NNS）时，Minkowski距离的性能最佳。

    The characteristics of data like distribution and heterogeneity, become more complex and counterintuitive as the dimensionality increases. This phenomenon is known as curse of dimensionality, where common patterns and relationships (e.g., internal and boundary pattern) that hold in low-dimensional space may be invalid in higher-dimensional space. It leads to a decreasing performance for the regression, classification or clustering models or algorithms. Curse of dimensionality can be attributed to many causes. In this paper, we first summarize five challenges associated with manipulating high-dimensional data, and explains the potential causes for the failure of regression, classification or clustering tasks. Subsequently, we delve into two major causes of the curse of dimensionality, distance concentration and manifold effect, by performing theoretical and empirical analyses. The results demonstrate that nearest neighbor search (NNS) using three typical distance measurements, Minkowski
    

