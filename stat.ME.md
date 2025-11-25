# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heterogeneous Treatment Effects and Causal Mechanisms](https://arxiv.org/abs/2404.01566) | 该论文讨论了异质性治疗效应对于因果机制的推断，并指出了存在的挑战和局限性。 |
| [^2] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |
| [^3] | [Minimising the Expected Posterior Entropy Yields Optimal Summary Statistics.](http://arxiv.org/abs/2206.02340) | 该论文介绍了从大型数据集中提取低维摘要统计量的重要性，提出了通过最小化后验熵来获取最优摘要统计量的方法，并提供了实践建议和示例验证。 |

# 详细

[^1]: 异质性治疗效应和因果机制

    Heterogeneous Treatment Effects and Causal Mechanisms

    [https://arxiv.org/abs/2404.01566](https://arxiv.org/abs/2404.01566)

    该论文讨论了异质性治疗效应对于因果机制的推断，并指出了存在的挑战和局限性。

    

    arXiv:2404.01566v1 公告类型:新摘要:可信度革命推动了利用研究设计来识别和估计因果效应的发展。然而，了解哪些机制产生了测量的因果效应仍然是一个挑战。目前关于定量评估机制的主要方法依赖于检测与处理前协变量相关的异质性处理效应。本文开发了一个框架，用于理解在何种情况下这种异质性处理效应的存在能够支持有关机制激活的推断。我们首先展示，这种设计在没有额外、通常是隐含的假设的情况下无法为机制激活提供证据。此外，即使这些假设得到满足，如果一个测量结果是通过直接受影响的感兴趣理论结果的非线性转换产生的，异质性处理效应也不足以推断机制激活。我们提供了n

    arXiv:2404.01566v1 Announce Type: new  Abstract: The credibility revolution advances the use of research designs that permit identification and estimation of causal effects. However, understanding which mechanisms produce measured causal effects remains a challenge. A dominant current approach to the quantitative evaluation of mechanisms relies on the detection of heterogeneous treatment effects with respect to pre-treatment covariates. This paper develops a framework to understand when the existence of such heterogeneous treatment effects can support inferences about the activation of a mechanism. We show first that this design cannot provide evidence of mechanism activation without additional, generally implicit, assumptions. Further, even when these assumptions are satisfied, if a measured outcome is produced by a non-linear transformation of a directly-affected outcome of theoretical interest, heterogeneous treatment effects are not informative of mechanism activation. We provide n
    
[^2]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    
[^3]: 最小化后验熵产生了最优摘要统计量

    Minimising the Expected Posterior Entropy Yields Optimal Summary Statistics. (arXiv:2206.02340v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2206.02340](http://arxiv.org/abs/2206.02340)

    该论文介绍了从大型数据集中提取低维摘要统计量的重要性，提出了通过最小化后验熵来获取最优摘要统计量的方法，并提供了实践建议和示例验证。

    

    从大型数据集中提取低维摘要统计量对于高效（无似然）推断非常重要。我们对不同类别的摘要进行了表征，并证明它们对于正确分析降维算法至关重要。我们建议通过在模型的先验预测分布下最小化期望后验熵（EPE）来获取摘要。许多现有方法等效于或是最小化EPE的特殊或极限情况。我们开发了一种方法来获取最小化EPE的高保真摘要；我们将其应用于基准和真实世界的示例。我们既提供了获取有效摘要的统一视角，又为实践者提供了具体建议。

    Extracting low-dimensional summary statistics from large datasets is essential for efficient (likelihood-free) inference. We characterise different classes of summaries and demonstrate their importance for correctly analysing dimensionality reduction algorithms. We propose obtaining summaries by minimising the expected posterior entropy (EPE) under the prior predictive distribution of the model. Many existing methods are equivalent to or are special or limiting cases of minimising the EPE. We develop a method to obtain high-fidelity summaries that minimise the EPE; we apply it to benchmark and real-world examples. We both offer a unifying perspective for obtaining informative summaries and provide concrete recommendations for practitioners.
    

