# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A testing-based approach to assess the clusterability of categorical data.](http://arxiv.org/abs/2307.07346) | 这项研究提出了一种基于测试的方法，用于评估分类数据的聚类性能。通过计算属性对的卡方统计量的和得到一个解析的p值作为评估指标。结果表明这种方法在基准分类数据集上的表现优于其他方法。 |

# 详细

[^1]: 一种基于测试的方法来评估分类数据的聚类性

    A testing-based approach to assess the clusterability of categorical data. (arXiv:2307.07346v1 [cs.LG])

    [http://arxiv.org/abs/2307.07346](http://arxiv.org/abs/2307.07346)

    这项研究提出了一种基于测试的方法，用于评估分类数据的聚类性能。通过计算属性对的卡方统计量的和得到一个解析的p值作为评估指标。结果表明这种方法在基准分类数据集上的表现优于其他方法。

    

    聚类性评估的目标是检查数据集中是否存在聚类结构。作为聚类分析中一个至关重要但常常被忽视的问题，我们在应用任何聚类算法之前都需要进行这样的检验。如果一个数据集不可聚类，则任何后续的聚类分析都不会产生有效结果。尽管它的重要性，但现有研究中的大部分关注数值数据，将对分类数据的聚类性评估问题留给成为一个未解决的问题。在这里，我们介绍了TestCat，一种基于测试的方法来评估分类数据的聚类性，使用解析的p值作为评估指标。TestCat的关键思想是，可聚类的分类数据具有许多强相关的属性对，因此将所有属性对的卡方统计量的和作为p值计算的测试统计量。我们将我们的方法应用于一组基准分类数据集，结果表明TestCat的表现优于现有的其他方法。

    The objective of clusterability evaluation is to check whether a clustering structure exists within the data set. As a crucial yet often-overlooked issue in cluster analysis, it is essential to conduct such a test before applying any clustering algorithm. If a data set is unclusterable, any subsequent clustering analysis would not yield valid results. Despite its importance, the majority of existing studies focus on numerical data, leaving the clusterability evaluation issue for categorical data as an open problem. Here we present TestCat, a testing-based approach to assess the clusterability of categorical data in terms of an analytical $p$-value. The key idea underlying TestCat is that clusterable categorical data possess many strongly correlated attribute pairs and hence the sum of chi-squared statistics of all attribute pairs is employed as the test statistic for $p$-value calculation. We apply our method to a set of benchmark categorical data sets, showing that TestCat outperforms
    

