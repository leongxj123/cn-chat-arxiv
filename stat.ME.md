# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^2] | [Scalable Estimation of Multinomial Response Models with Uncertain Consideration Sets.](http://arxiv.org/abs/2308.12470) | 这篇论文提出了一种克服估计多项式响应模型中指数级支持问题的方法，通过使用基于列联表概率分布的考虑集概率模型。 |

# 详细

[^1]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^2]: 可伸缩估计具有不确定的选项集的多项式响应模型

    Scalable Estimation of Multinomial Response Models with Uncertain Consideration Sets. (arXiv:2308.12470v1 [stat.ME])

    [http://arxiv.org/abs/2308.12470](http://arxiv.org/abs/2308.12470)

    这篇论文提出了一种克服估计多项式响应模型中指数级支持问题的方法，通过使用基于列联表概率分布的考虑集概率模型。

    

    在交叉或纵向数据的无序多项式响应模型拟合中的一个标准假设是，响应来自于相同的J个类别集合。然而，当响应度量主体做出的选择时，更适合假设多项式响应的分布是在主体特定的考虑集条件下，其中这个考虑集是从{1,2, ..., J}的幂集中抽取的。由于这个幂集的基数在J中是指数级的，一般来说估计是无法实现的。在本文中，我们提供了一种克服这个问题的方法。这种方法中的一个关键步骤是基于在列联表上的概率分布的一般表示的考虑集的概率模型。尽管这个分布的支持是指数级大的，但给定参数的考虑集的后验分布通常是稀疏的。

    A standard assumption in the fitting of unordered multinomial response models for J mutually exclusive nominal categories, on cross-sectional or longitudinal data, is that the responses arise from the same set of J categories between subjects. However, when responses measure a choice made by the subject, it is more appropriate to assume that the distribution of multinomial responses is conditioned on a subject-specific consideration set, where this consideration set is drawn from the power set of {1,2,...,J}. Because the cardinality of this power set is exponential in J, estimation is infeasible in general. In this paper, we provide an approach to overcoming this problem. A key step in the approach is a probability model over consideration sets, based on a general representation of probability distributions on contingency tables. Although the support of this distribution is exponentially large, the posterior distribution over consideration sets given parameters is typically sparse, and
    

