# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Debiasing and a local analysis for population clustering using semidefinite programming.](http://arxiv.org/abs/2401.10927) | 本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。 |
| [^2] | [A powerful rank-based correction to multiple testing under positive dependency.](http://arxiv.org/abs/2311.10900) | 我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。 |

# 详细

[^1]: 使用半正定规划的去偏和局部分析进行人群聚类

    Debiasing and a local analysis for population clustering using semidefinite programming. (arXiv:2401.10927v1 [stat.ML])

    [http://arxiv.org/abs/2401.10927](http://arxiv.org/abs/2401.10927)

    本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。

    

    本文考虑了从混合的2个次高斯分布中抽取的小数据样本的分区问题。我们分析了同一作者提出的计算高效的算法，将数据根据其原始种群大致分为两组，给定一个小样本。本文的研究动机是将个体根据其原始种群使用p个标记进行聚类，当任意两个种群之间的差异很小时。我们基于整数二次规划的半正定松弛形式构建，该规划问题本质上是在一个图上找到最大割，其中割中的边权重表示基于它们的p个特征的两个节点之间的不相似度得分。我们用Δ^2:=pγ来表示两个中心（均值向量）之间的ℓ_2^2距离，即μ^(1), μ^(2)∈ℝ^p。目标是在交换精度和计算效率之间提供全面的权衡。

    In this paper, we consider the problem of partitioning a small data sample of size $n$ drawn from a mixture of $2$ sub-gaussian distributions. In particular, we analyze computational efficient algorithms proposed by the same author, to partition data into two groups approximately according to their population of origin given a small sample. This work is motivated by the application of clustering individuals according to their population of origin using $p$ markers, when the divergence between any two of the populations is small. We build upon the semidefinite relaxation of an integer quadratic program that is formulated essentially as finding the maximum cut on a graph, where edge weights in the cut represent dissimilarity scores between two nodes based on their $p$ features. Here we use $\Delta^2 :=p \gamma$ to denote the $\ell_2^2$ distance between two centers (mean vectors), namely, $\mu^{(1)}$, $\mu^{(2)}$ $\in$ $\mathbb{R}^p$. The goal is to allow a full range of tradeoffs between
    
[^2]: 一种基于秩的多重检验正相关依赖的强大修正方法

    A powerful rank-based correction to multiple testing under positive dependency. (arXiv:2311.10900v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.10900](http://arxiv.org/abs/2311.10900)

    我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。

    

    我们开发了一种能够高效利用可能相关的统计假设检验之间正相关性的家族误差率(FWER)控制的新型多重假设检验修正算法$\texttt{max-rank}$。我们的方法概念上很直观，依赖于在计算的统计检验的秩域使用$\max$算子。通过理论和经验的比较，我们证明了在存在正相关依赖的情况下，我们的方法优于经常使用的Bonferroni修正，而在不存在正相关依赖的情况下等效。我们的优势随着测试数量的增加而增加，同时在保证FWER控制的情况下保持高统计功效。我们特别将我们的算法应用于并行置换检验的背景中，这是在我们主要应用的一种复杂预测场景中产生的情况下。

    We develop a novel multiple hypothesis testing correction with family-wise error rate (FWER) control that efficiently exploits positive dependencies between potentially correlated statistical hypothesis tests. Our proposed algorithm $\texttt{max-rank}$ is conceptually straight-forward, relying on the use of a $\max$-operator in the rank domain of computed test statistics. We compare our approach to the frequently employed Bonferroni correction, theoretically and empirically demonstrating its superiority over Bonferroni in the case of existing positive dependency, and its equivalence otherwise. Our advantage over Bonferroni increases as the number of tests rises, and we maintain high statistical power whilst ensuring FWER control. We specifically frame our algorithm in the context of parallel permutation testing, a scenario that arises in our primary application of conformal prediction, a recently popularized approach for quantifying uncertainty in complex predictive settings.
    

