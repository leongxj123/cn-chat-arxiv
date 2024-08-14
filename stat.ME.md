# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [flexBART: Flexible Bayesian regression trees with categorical predictors.](http://arxiv.org/abs/2211.04459) | 本论文提出了一种新的灵活贝叶斯回归树模型flexBART，可以在划分分类水平的离散集时，将多个水平分配给决策树节点的两个分支，从而实现了对分类预测变量的灵活建模，跨级别的数据部分汇总能力也得到了改进。 |

# 详细

[^1]: flexBART:具有分类预测变量的灵活贝叶斯回归树

    flexBART: Flexible Bayesian regression trees with categorical predictors. (arXiv:2211.04459v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2211.04459](http://arxiv.org/abs/2211.04459)

    本论文提出了一种新的灵活贝叶斯回归树模型flexBART，可以在划分分类水平的离散集时，将多个水平分配给决策树节点的两个分支，从而实现了对分类预测变量的灵活建模，跨级别的数据部分汇总能力也得到了改进。

    

    大多数贝叶斯加法回归树（BART）的实现方法采用独热编码将分类预测变量替换为多个二进制指标，每个指标对应于每个级别或类别。用这些指标构建的回归树通过反复删除一个级别来划分分类水平的离散集。然而，绝大多数分割不能使用此策略构建，严重限制了BART在跨级别的数据部分汇总方面的能力。受对棒球数据和邻里犯罪动态分析的启发，我们通过重新实现以能够将多个水平分配给决策树节点的两个分支的回归树来克服了这个限制。为了对聚合为小区域的空间数据建模，我们进一步提出了一个新的决策规则先验，通过从适当定义的网络的随机生成树中删除一个随机边来创建空间连续的区域。我们的重新实现，可在R的flexBART软件包中使用，允许灵活地建模分类预测变量并改进跨不同级别的数据部分汇总。

    Most implementations of Bayesian additive regression trees (BART) one-hot encode categorical predictors, replacing each one with several binary indicators, one for every level or category. Regression trees built with these indicators partition the discrete set of categorical levels by repeatedly removing one level at a time. Unfortunately, the vast majority of partitions cannot be built with this strategy, severely limiting BART's ability to partially pool data across groups of levels. Motivated by analyses of baseball data and neighborhood-level crime dynamics, we overcame this limitation by re-implementing BART with regression trees that can assign multiple levels to both branches of a decision tree node. To model spatial data aggregated into small regions, we further proposed a new decision rule prior that creates spatially contiguous regions by deleting a random edge from a random spanning tree of a suitably defined network. Our re-implementation, which is available in the flexBART
    

