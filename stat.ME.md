# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Assessing Heterogeneity of Treatment Effects.](http://arxiv.org/abs/2306.15048) | 该论文介绍了一种评估治疗效果异质性的方法，通过使用治疗组和对照组结果的分位数范围，即使平均效果不显著，也可以提供有用的信息。 |
| [^2] | [Locally Adaptive Algorithms for Multiple Testing with Network Structure, with Application to Genome-Wide Association Studies.](http://arxiv.org/abs/2203.11461) | 本文提出了一种基于网络结构的局部自适应结构学习算法，可将LD网络数据和多个样本的辅助数据整合起来，通过数据驱动的权重分配方法实现对多重检验的控制，并在网络数据具有信息量时具有更高的功效。 |

# 详细

[^1]: 评估治疗效果的异质性

    Assessing Heterogeneity of Treatment Effects. (arXiv:2306.15048v1 [econ.EM])

    [http://arxiv.org/abs/2306.15048](http://arxiv.org/abs/2306.15048)

    该论文介绍了一种评估治疗效果异质性的方法，通过使用治疗组和对照组结果的分位数范围，即使平均效果不显著，也可以提供有用的信息。

    

    异质性治疗效果在经济学中非常重要，但是其评估常常受到个体治疗效果无法确定的困扰。例如，我们可能希望评估保险对本来不健康的人的健康影响，但是只给不健康的人买保险是不可行的，因此这些人的因果效应无法确定。又或者，我们可能对最低工资上涨中赢家的份额感兴趣，但是在没有观察到反事实的情况下，赢家也无法确定。这种异质性常常通过分位数治疗效果来评估，但其解释并不清晰，结论有时也不一致。我们展示了通过治疗组和对照组结果的分位数，这些数值范围是可以确定的，即使平均治疗效果并不显著，它们仍然可以提供有用信息。两个应用实例展示了这些范围如何帮助我们了解治疗效果的异质性。

    Treatment effect heterogeneity is of major interest in economics, but its assessment is often hindered by the fundamental lack of identification of the individual treatment effects. For example, we may want to assess the effect of insurance on the health of otherwise unhealthy individuals, but it is infeasible to insure only the unhealthy, and thus the causal effects for those are not identified. Or, we may be interested in the shares of winners from a minimum wage increase, while without observing the counterfactual, the winners are not identified. Such heterogeneity is often assessed by quantile treatment effects, which do not come with clear interpretation and the takeaway can sometimes be equivocal. We show that, with the quantiles of the treated and control outcomes, the ranges of these quantities are identified and can be informative even when the average treatment effects are not significant. Two applications illustrate how these ranges can inform us about heterogeneity of the t
    
[^2]: 基于网络结构的局部自适应多重检验算法，及其在全基因组关联研究中的应用

    Locally Adaptive Algorithms for Multiple Testing with Network Structure, with Application to Genome-Wide Association Studies. (arXiv:2203.11461v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2203.11461](http://arxiv.org/abs/2203.11461)

    本文提出了一种基于网络结构的局部自适应结构学习算法，可将LD网络数据和多个样本的辅助数据整合起来，通过数据驱动的权重分配方法实现对多重检验的控制，并在网络数据具有信息量时具有更高的功效。

    

    链接分析在全基因组关联研究中起着重要作用，特别是在揭示与疾病表型相关的连锁不平衡（LD）的SNP共同影响方面。然而，LD网络数据的潜力在文献中往往被忽视或未充分利用。在本文中，我们提出了一个局部自适应结构学习算法（LASLA），为整合网络数据或来自相关源域的多个样本的辅助数据提供了一个有原则且通用的框架；可能具有不同的维度/结构和不同的人群。LASLA采用$p$值加权方法，利用结构洞察力为各个检验点分配数据驱动的权重。理论分析表明，当主要统计量独立或弱相关时，LASLA可以渐近地控制FDR，并在网络数据具有信息量时实现更高的功效。通过各种合成实验和一个应用案例，展示了LASLA的效率。

    Linkage analysis has provided valuable insights to the GWAS studies, particularly in revealing that SNPs in linkage disequilibrium (LD) can jointly influence disease phenotypes. However, the potential of LD network data has often been overlooked or underutilized in the literature. In this paper, we propose a locally adaptive structure learning algorithm (LASLA) that provides a principled and generic framework for incorporating network data or multiple samples of auxiliary data from related source domains; possibly in different dimensions/structures and from diverse populations. LASLA employs a $p$-value weighting approach, utilizing structural insights to assign data-driven weights to individual test points. Theoretical analysis shows that LASLA can asymptotically control FDR with independent or weakly dependent primary statistics, and achieve higher power when the network data is informative. Efficiency again of LASLA is illustrated through various synthetic experiments and an applica
    

