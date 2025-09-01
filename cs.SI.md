# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WGoM: A novel model for categorical data with weighted responses.](http://arxiv.org/abs/2310.10989) | 本文提出了一种名为加权成员级别（WGoM）模型，用于解决基于分类数据的潜在类别推断问题。与现有模型相比，WGoM更通用且适用于具有连续或负响应的数据集。通过提出的算法，我们能够准确高效地估计潜在混合成员和其他WGoM参数，并且通过实验证明了该算法的性能和实用潜力。 |
| [^2] | [Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model.](http://arxiv.org/abs/2306.12968) | 本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。 |

# 详细

[^1]: WGoM：一种适用于带加权响应的分类数据的新模型

    WGoM: A novel model for categorical data with weighted responses. (arXiv:2310.10989v1 [cs.SI])

    [http://arxiv.org/abs/2310.10989](http://arxiv.org/abs/2310.10989)

    本文提出了一种名为加权成员级别（WGoM）模型，用于解决基于分类数据的潜在类别推断问题。与现有模型相比，WGoM更通用且适用于具有连续或负响应的数据集。通过提出的算法，我们能够准确高效地估计潜在混合成员和其他WGoM参数，并且通过实验证明了该算法的性能和实用潜力。

    

    Graded of Membership（GoM）模型是一种用于推断分类数据中潜在类别的强大工具，使得个体可以属于多个潜在类别。然而，该模型仅适用于具有非负整数响应的分类数据，使得它无法应用于具有连续或负响应的数据集。为了解决这个限制，本文提出了一种名为加权成员级别（WGoM）模型的新模型。与GoM相比，我们的WGoM在响应矩阵的生成上放宽了GoM的分布约束，并且比GoM更通用。我们还提出了一种估计潜在混合成员和其他WGoM参数的算法。我们推导了估计参数的误差界限，并且证明了算法的统计一致性。该算法的性能在合成和真实世界的数据集中得到了验证。结果表明我们的算法准确高效，具有很高的实用潜力。

    The Graded of Membership (GoM) model is a powerful tool for inferring latent classes in categorical data, which enables subjects to belong to multiple latent classes. However, its application is limited to categorical data with nonnegative integer responses, making it inappropriate for datasets with continuous or negative responses. To address this limitation, this paper proposes a novel model named the Weighted Grade of Membership (WGoM) model. Compared with GoM, our WGoM relaxes GoM's distribution constraint on the generation of a response matrix and it is more general than GoM. We then propose an algorithm to estimate the latent mixed memberships and the other WGoM parameters. We derive the error bounds of the estimated parameters and show that the algorithm is statistically consistent. The algorithmic performance is validated in both synthetic and real-world datasets. The results demonstrate that our algorithm is accurate and efficient, indicating its high potential for practical a
    
[^2]: 标记随机块模型中的最优簇恢复问题

    Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model. (arXiv:2306.12968v1 [cs.SI])

    [http://arxiv.org/abs/2306.12968](http://arxiv.org/abs/2306.12968)

    本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。

    

    本文考虑在有限数量的簇的情况下，用标记随机块模型（LSBM）恢复隐藏的社群，其中簇大小随着物品总数$n$的增长而线性增长。在LSBM中，为每对物品（独立地）观测到一个标签。我们的目标是设计一种有效的算法，利用观测到的标签来恢复簇。为此，我们重新审视了关于期望被任何聚类算法误分类的物品数量的实例特定下界。我们提出了实例自适应聚类（IAC），这是第一个在期望和高概率下都能匹配这些下界表现的算法。IAC由一次谱聚类算法和一个迭代的基于似然的簇分配改进组成。这种方法基于实例特定的下界，不需要任何模型参数，包括簇的数量。通过仅执行一次谱聚类，IAC在计算和存储方面都是高效的。

    We consider the problem of recovering hidden communities in the Labeled Stochastic Block Model (LSBM) with a finite number of clusters, where cluster sizes grow linearly with the total number $n$ of items. In the LSBM, a label is (independently) observed for each pair of items. Our objective is to devise an efficient algorithm that recovers clusters using the observed labels. To this end, we revisit instance-specific lower bounds on the expected number of misclassified items satisfied by any clustering algorithm. We present Instance-Adaptive Clustering (IAC), the first algorithm whose performance matches these lower bounds both in expectation and with high probability. IAC consists of a one-time spectral clustering algorithm followed by an iterative likelihood-based cluster assignment improvement. This approach is based on the instance-specific lower bound and does not require any model parameters, including the number of clusters. By performing the spectral clustering only once, IAC m
    

