# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Computing Optimal Tree Ensembles.](http://arxiv.org/abs/2306.04423) | 该论文提出了两种新算法以计算旨在各种度量方面最优的决策树集合，并且引入了“证明树技术”来大大改进可处理性结果。 |
| [^2] | [Subsampling Suffices for Adaptive Data Analysis.](http://arxiv.org/abs/2302.08661) | 子采样是自适应数据分析中的关键方法，仅需基于随机子样本和少量比特输出的查询，即可保证代表性和泛化性。 |

# 详细

[^1]: 计算最优树集的方法

    On Computing Optimal Tree Ensembles. (arXiv:2306.04423v1 [cs.LG])

    [http://arxiv.org/abs/2306.04423](http://arxiv.org/abs/2306.04423)

    该论文提出了两种新算法以计算旨在各种度量方面最优的决策树集合，并且引入了“证明树技术”来大大改进可处理性结果。

    

    随机森林和决策树集合是分类和回归的广泛应用方法。最近的算法进展允许计算旨在各种度量方面（如大小或深度）最优的决策树。我们不知道有关树集合的此类研究，并旨在为该领域做出贡献。主要的是，我们提供了两种新算法和相应的下限。首先，我们能够转移和大大改进决策树的可处理性结果，获得一个 $(6\delta D S)^S \cdot poly$-time 算法，其中 $S$ 是树集合中割数，$D$ 是最大域大小，$\delta$ 是两个示例之间存在不同特征的最大数量。为了达到这个目的，我们引入了证明树技术，这似乎对实践也很有前途。其次，我们表明，动态规划对于决策树已经取得了成功，而对于树集合也可能是可行的，提供了一个 $\ell^n \cdot poly$-t。

    Random forests and, more generally, (decision\nobreakdash-)tree ensembles are widely used methods for classification and regression. Recent algorithmic advances allow to compute decision trees that are optimal for various measures such as their size or depth. We are not aware of such research for tree ensembles and aim to contribute to this area. Mainly, we provide two novel algorithms and corresponding lower bounds. First, we are able to carry over and substantially improve on tractability results for decision trees, obtaining a $(6\delta D S)^S \cdot poly$-time algorithm, where $S$ is the number of cuts in the tree ensemble, $D$ the largest domain size, and $\delta$ is the largest number of features in which two examples differ. To achieve this, we introduce the witness-tree technique which also seems promising for practice. Second, we show that dynamic programming, which has been successful for decision trees, may also be viable for tree ensembles, providing an $\ell^n \cdot poly$-t
    
[^2]: 自适应数据分析中的子采样足够

    Subsampling Suffices for Adaptive Data Analysis. (arXiv:2302.08661v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.08661](http://arxiv.org/abs/2302.08661)

    子采样是自适应数据分析中的关键方法，仅需基于随机子样本和少量比特输出的查询，即可保证代表性和泛化性。

    

    确保对数据集的分析代表整个样本总体是统计学中的核心问题之一。大多数经典技术假设数据集与分析师的查询无关，并在多次、自适应选择的查询中失效。这个“自适应数据分析”问题在Dwork等人（STOC，2015）和Hardt和Ullman（FOCS，2014）的开创性工作中得到了形式化。我们确定了一个非常简单的假设集，使得即使在自适应选择的情况下，查询仍然具有代表性：唯一的要求是每个查询采用随机子样本作为输入并输出少量比特。这个结果表明，子采样中固有的噪音足以保证查询的响应具有泛化性。这种基于子采样的框架的简单性使其能够模拟之前研究所未涵盖的各种实际情境。

    Ensuring that analyses performed on a dataset are representative of the entire population is one of the central problems in statistics. Most classical techniques assume that the dataset is independent of the analyst's query and break down in the common setting where a dataset is reused for multiple, adaptively chosen, queries. This problem of \emph{adaptive data analysis} was formalized in the seminal works of Dwork et al. (STOC, 2015) and Hardt and Ullman (FOCS, 2014).  We identify a remarkably simple set of assumptions under which the queries will continue to be representative even when chosen adaptively: The only requirements are that each query takes as input a random subsample and outputs few bits. This result shows that the noise inherent in subsampling is sufficient to guarantee that query responses generalize. The simplicity of this subsampling-based framework allows it to model a variety of real-world scenarios not covered by prior work.  In addition to its simplicity, we demo
    

