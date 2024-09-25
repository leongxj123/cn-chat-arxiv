# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Subsampling Suffices for Adaptive Data Analysis.](http://arxiv.org/abs/2302.08661) | 子采样是自适应数据分析中的关键方法，仅需基于随机子样本和少量比特输出的查询，即可保证代表性和泛化性。 |

# 详细

[^1]: 自适应数据分析中的子采样足够

    Subsampling Suffices for Adaptive Data Analysis. (arXiv:2302.08661v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.08661](http://arxiv.org/abs/2302.08661)

    子采样是自适应数据分析中的关键方法，仅需基于随机子样本和少量比特输出的查询，即可保证代表性和泛化性。

    

    确保对数据集的分析代表整个样本总体是统计学中的核心问题之一。大多数经典技术假设数据集与分析师的查询无关，并在多次、自适应选择的查询中失效。这个“自适应数据分析”问题在Dwork等人（STOC，2015）和Hardt和Ullman（FOCS，2014）的开创性工作中得到了形式化。我们确定了一个非常简单的假设集，使得即使在自适应选择的情况下，查询仍然具有代表性：唯一的要求是每个查询采用随机子样本作为输入并输出少量比特。这个结果表明，子采样中固有的噪音足以保证查询的响应具有泛化性。这种基于子采样的框架的简单性使其能够模拟之前研究所未涵盖的各种实际情境。

    Ensuring that analyses performed on a dataset are representative of the entire population is one of the central problems in statistics. Most classical techniques assume that the dataset is independent of the analyst's query and break down in the common setting where a dataset is reused for multiple, adaptively chosen, queries. This problem of \emph{adaptive data analysis} was formalized in the seminal works of Dwork et al. (STOC, 2015) and Hardt and Ullman (FOCS, 2014).  We identify a remarkably simple set of assumptions under which the queries will continue to be representative even when chosen adaptively: The only requirements are that each query takes as input a random subsample and outputs few bits. This result shows that the noise inherent in subsampling is sufficient to guarantee that query responses generalize. The simplicity of this subsampling-based framework allows it to model a variety of real-world scenarios not covered by prior work.  In addition to its simplicity, we demo
    

