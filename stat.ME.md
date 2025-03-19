# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A powerful rank-based correction to multiple testing under positive dependency.](http://arxiv.org/abs/2311.10900) | 我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。 |

# 详细

[^1]: 一种基于秩的多重检验正相关依赖的强大修正方法

    A powerful rank-based correction to multiple testing under positive dependency. (arXiv:2311.10900v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.10900](http://arxiv.org/abs/2311.10900)

    我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。

    

    我们开发了一种能够高效利用可能相关的统计假设检验之间正相关性的家族误差率(FWER)控制的新型多重假设检验修正算法$\texttt{max-rank}$。我们的方法概念上很直观，依赖于在计算的统计检验的秩域使用$\max$算子。通过理论和经验的比较，我们证明了在存在正相关依赖的情况下，我们的方法优于经常使用的Bonferroni修正，而在不存在正相关依赖的情况下等效。我们的优势随着测试数量的增加而增加，同时在保证FWER控制的情况下保持高统计功效。我们特别将我们的算法应用于并行置换检验的背景中，这是在我们主要应用的一种复杂预测场景中产生的情况下。

    We develop a novel multiple hypothesis testing correction with family-wise error rate (FWER) control that efficiently exploits positive dependencies between potentially correlated statistical hypothesis tests. Our proposed algorithm $\texttt{max-rank}$ is conceptually straight-forward, relying on the use of a $\max$-operator in the rank domain of computed test statistics. We compare our approach to the frequently employed Bonferroni correction, theoretically and empirically demonstrating its superiority over Bonferroni in the case of existing positive dependency, and its equivalence otherwise. Our advantage over Bonferroni increases as the number of tests rises, and we maintain high statistical power whilst ensuring FWER control. We specifically frame our algorithm in the context of parallel permutation testing, a scenario that arises in our primary application of conformal prediction, a recently popularized approach for quantifying uncertainty in complex predictive settings.
    

