# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diversity-aware clustering: Computational Complexity and Approximation Algorithms.](http://arxiv.org/abs/2401.05502) | 本研究讨论了多样性感知聚类问题，在选择聚类中心时要考虑多个属性，同时最小化聚类目标。我们提出了针对不同聚类目标的参数化近似算法，这些算法在保证聚类质量的同时，具有紧确的近似比。 |

# 详细

[^1]: 多样性感知聚类：计算复杂性和近似算法

    Diversity-aware clustering: Computational Complexity and Approximation Algorithms. (arXiv:2401.05502v1 [cs.DS])

    [http://arxiv.org/abs/2401.05502](http://arxiv.org/abs/2401.05502)

    本研究讨论了多样性感知聚类问题，在选择聚类中心时要考虑多个属性，同时最小化聚类目标。我们提出了针对不同聚类目标的参数化近似算法，这些算法在保证聚类质量的同时，具有紧确的近似比。

    

    在这项工作中，我们研究了多样性感知聚类问题，其中数据点与多个属性相关联，形成交叉的组。聚类解决方案需要确保从每个组中选择最少数量的聚类中心，同时最小化聚类目标，可以是$k$-中位数，$k$-均值或$k$-供应商。我们提出了参数化近似算法，近似比分别为$1+\frac{2}{e}$，$1+\frac{8}{e}$和$3$，用于多样性感知$k$-中位数，多样性感知$k$-均值和多样性感知$k$-供应商。这些近似比在假设Gap-ETH和FPT $\neq$ W[2]的情况下是紧确的。对于公平$k$-中位数和公平$k$-均值的不相交工厂组，我们提出了参数化近似算法，近似比分别为$1+\frac{2}{e}$和$1+\frac{8}{e}$。对于具有不相交工厂组的公平$k$-供应商，我们提出了一个多项式时间近似算法，因子为$3$。

    In this work, we study diversity-aware clustering problems where the data points are associated with multiple attributes resulting in intersecting groups. A clustering solution need to ensure that a minimum number of cluster centers are chosen from each group while simultaneously minimizing the clustering objective, which can be either $k$-median, $k$-means or $k$-supplier. We present parameterized approximation algorithms with approximation ratios $1+ \frac{2}{e}$, $1+\frac{8}{e}$ and $3$ for diversity-aware $k$-median, diversity-aware $k$-means and diversity-aware $k$-supplier, respectively. The approximation ratios are tight assuming Gap-ETH and FPT $\neq$ W[2]. For fair $k$-median and fair $k$-means with disjoint faicility groups, we present parameterized approximation algorithm with approximation ratios $1+\frac{2}{e}$ and $1+\frac{8}{e}$, respectively. For fair $k$-supplier with disjoint facility groups, we present a polynomial-time approximation algorithm with factor $3$, improv
    

