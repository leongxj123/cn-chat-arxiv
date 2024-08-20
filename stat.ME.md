# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reducing the dimensionality and granularity in hierarchical categorical variables](https://arxiv.org/abs/2403.03613) | 提出了一种减少层次分类变量维度和粒度的方法，通过实体嵌入和自上而下聚类算法来降低层内维度和整体粒度。 |
| [^2] | [Off-policy Evaluation in Doubly Inhomogeneous Environments.](http://arxiv.org/abs/2306.08719) | 本研究在双非均匀环境下研究了离线策略评估(OPE)，提出了一种潜在因子模型用于奖励和观测转移函数，并开发了一个通用的OPE框架。该研究对标准RL假设不满足的环境中的OPE做出了理论贡献，并提供了几种实用的方法。 |
| [^3] | [Estimating large causal polytrees from small samples.](http://arxiv.org/abs/2209.07028) | 本文介绍了一种算法，可以在变量数量远大于样本大小的情况下，准确地估计大规模因果多树结构，而几乎不需要任何分布或建模的假设。 |

# 详细

[^1]: 减少层次分类变量的维度和粒度

    Reducing the dimensionality and granularity in hierarchical categorical variables

    [https://arxiv.org/abs/2403.03613](https://arxiv.org/abs/2403.03613)

    提出了一种减少层次分类变量维度和粒度的方法，通过实体嵌入和自上而下聚类算法来降低层内维度和整体粒度。

    

    层次分类变量往往具有许多级别（高粒度）和每个级别内许多类别（高维度）。将这些协变量包含在预测模型中可能导致过度拟合和估计问题。在当前文献中，层次协变量通常通过嵌套随机效应来纳入。然而，这并不有助于假设类别对响应变量具有相同的影响。本文提出了一种获得层次分类变量简化表示的方法。我们展示了如何在层次设置中应用实体嵌入。随后，我们提出了一种自上而下的聚类算法，利用嵌入中编码的信息来减少层内维度以及层次分类变量的整体粒度。在模拟实验中，我们展示了我们的方法可以有效地应用。

    arXiv:2403.03613v1 Announce Type: cross  Abstract: Hierarchical categorical variables often exhibit many levels (high granularity) and many classes within each level (high dimensionality). This may cause overfitting and estimation issues when including such covariates in a predictive model. In current literature, a hierarchical covariate is often incorporated via nested random effects. However, this does not facilitate the assumption of classes having the same effect on the response variable. In this paper, we propose a methodology to obtain a reduced representation of a hierarchical categorical variable. We show how entity embedding can be applied in a hierarchical setting. Subsequently, we propose a top-down clustering algorithm which leverages the information encoded in the embeddings to reduce both the within-level dimensionality as well as the overall granularity of the hierarchical categorical variable. In simulation experiments, we show that our methodology can effectively appro
    
[^2]: 双非均匀环境下的离线策略评估

    Off-policy Evaluation in Doubly Inhomogeneous Environments. (arXiv:2306.08719v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2306.08719](http://arxiv.org/abs/2306.08719)

    本研究在双非均匀环境下研究了离线策略评估(OPE)，提出了一种潜在因子模型用于奖励和观测转移函数，并开发了一个通用的OPE框架。该研究对标准RL假设不满足的环境中的OPE做出了理论贡献，并提供了几种实用的方法。

    

    本研究旨在研究离线策略评估（OPE）在两个关键的强化学习（RL）假设——时间稳定性和个体均匀性均被破坏的情况下的应用。为了处理“双非均匀性”，我们提出了一类潜在因子模型用于奖励和观测转移函数，并在此基础上开发了一个包含模型驱动和模型自由方法的通用OPE框架。据我们所知，这是第一篇在离线RL中开发统计上可靠的OPE方法的论文，并且涉及了标准RL假设不满足的环境。该研究深入理解了标准RL假设不满足的环境中的OPE，并在这些设置中提供了几种实用的方法。我们确定了所提出的价值估计器的理论性质，并通过实证研究表明我们的方法优于忽视时间非稳定性或个体异质性的竞争方法。最后，我们在一个数据集上说明了我们的方法。

    This work aims to study off-policy evaluation (OPE) under scenarios where two key reinforcement learning (RL) assumptions -- temporal stationarity and individual homogeneity are both violated. To handle the ``double inhomogeneities", we propose a class of latent factor models for the reward and observation transition functions, under which we develop a general OPE framework that consists of both model-based and model-free approaches. To our knowledge, this is the first paper that develops statistically sound OPE methods in offline RL with double inhomogeneities. It contributes to a deeper understanding of OPE in environments, where standard RL assumptions are not met, and provides several practical approaches in these settings. We establish the theoretical properties of the proposed value estimators and empirically show that our approach outperforms competing methods that ignore either temporal nonstationarity or individual heterogeneity. Finally, we illustrate our method on a data set
    
[^3]: 从小样本中估计大的因果多树

    Estimating large causal polytrees from small samples. (arXiv:2209.07028v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2209.07028](http://arxiv.org/abs/2209.07028)

    本文介绍了一种算法，可以在变量数量远大于样本大小的情况下，准确地估计大规模因果多树结构，而几乎不需要任何分布或建模的假设。

    

    我们考虑从相对较小的独立同分布样本中估计大的因果多树的问题。这是在变量数量与样本大小相比非常大的情况下确定因果结构的问题，例如基因调控网络。我们提出了一种算法，在这种情况下以高准确度恢复树形结构。该算法除了一些温和的非退化条件外，基本不需要分布或建模的假设。

    We consider the problem of estimating a large causal polytree from a relatively small i.i.d. sample. This is motivated by the problem of determining causal structure when the number of variables is very large compared to the sample size, such as in gene regulatory networks. We give an algorithm that recovers the tree with high accuracy in such settings. The algorithm works under essentially no distributional or modeling assumptions other than some mild non-degeneracy conditions.
    

