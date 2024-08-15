# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Improved Semi-parametric Bounds for Tail Probability and Expected Loss](https://arxiv.org/abs/2404.02400) | 本研究提出了对累积随机实现的尾部概率和期望线性损失的新的更尖锐界限，这些界限在基础分布半参数的情况下不受限制，补充了已有的结果，开辟了丰富的实际应用。 |
| [^2] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^3] | [Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec.](http://arxiv.org/abs/2310.17712) | 本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。 |

# 详细

[^1]: 关于累积随机实现的尾部概率和期望损失的改进半参数界限

    On Improved Semi-parametric Bounds for Tail Probability and Expected Loss

    [https://arxiv.org/abs/2404.02400](https://arxiv.org/abs/2404.02400)

    本研究提出了对累积随机实现的尾部概率和期望线性损失的新的更尖锐界限，这些界限在基础分布半参数的情况下不受限制，补充了已有的结果，开辟了丰富的实际应用。

    

    我们重新审视了当个别实现是独立的时，累积随机实现的尾部行为的基本问题，并在半参数的基础分布未受限制的情况下，开发了对尾部概率和期望线性损失的新的更尖锐的界限。我们的尖锐界限很好地补充了文献中已经建立的结果，包括基于聚合的方法，后者经常未能充分考虑独立性并使用不够优雅的证明。新的见解包括在非相同情况下的证明，达到界限的分布具有相等的范围属性，并且每个随机变量对总和的期望值的影响可以通过对Korkine恒等式的推广来孤立出来。我们表明，新的界限不仅补充了现有结果，而且开拓了大量的实际应用，包括改进定价。

    arXiv:2404.02400v1 Announce Type: new  Abstract: We revisit the fundamental issue of tail behavior of accumulated random realizations when individual realizations are independent, and we develop new sharper bounds on the tail probability and expected linear loss. The underlying distribution is semi-parametric in the sense that it remains unrestricted other than the assumed mean and variance. Our sharp bounds complement well-established results in the literature, including those based on aggregation, which often fail to take full account of independence and use less elegant proofs. New insights include a proof that in the non-identical case, the distributions attaining the bounds have the equal range property, and that the impact of each random variable on the expected value of the sum can be isolated using an extension of the Korkine identity. We show that the new bounds not only complement the extant results but also open up abundant practical applications, including improved pricing 
    
[^2]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^3]: 使用Node2Vec学习到的嵌入进行社区检测和分类的保证

    Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec. (arXiv:2310.17712v1 [stat.ML])

    [http://arxiv.org/abs/2310.17712](http://arxiv.org/abs/2310.17712)

    本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。

    

    将大型网络的节点嵌入到欧几里得空间中是现代机器学习中的常见目标，有各种工具可用。这些嵌入可以用作社区检测/节点聚类或链接预测等任务的特征，其性能达到了最先进水平。除了谱聚类方法之外，对于其他常用的学习嵌入方法，缺乏理论上的理解。在这项工作中，我们考察了由node2vec学习到的嵌入的理论属性。我们的主要结果表明，对node2vec生成的嵌入向量应用k-means聚类可以对（经过度修正的）随机块模型中的节点进行弱一致的社区恢复。我们还讨论了这些嵌入在节点和链接预测任务中的应用。我们通过实验证明了这个结果，并研究了它与网络数据的其他嵌入工具之间的关系。

    Embedding the nodes of a large network into an Euclidean space is a common objective in modern machine learning, with a variety of tools available. These embeddings can then be used as features for tasks such as community detection/node clustering or link prediction, where they achieve state of the art performance. With the exception of spectral clustering methods, there is little theoretical understanding for other commonly used approaches to learning embeddings. In this work we examine the theoretical properties of the embeddings learned by node2vec. Our main result shows that the use of k-means clustering on the embedding vectors produced by node2vec gives weakly consistent community recovery for the nodes in (degree corrected) stochastic block models. We also discuss the use of these embeddings for node and link prediction tasks. We demonstrate this result empirically, and examine how this relates to other embedding tools for network data.
    

