# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially private low-dimensional representation of high-dimensional data.](http://arxiv.org/abs/2305.17148) | 本文提出了一种在保护个人敏感信息的情况下，生成高效低维合成数据的算法，并在Wasserstein距离方面具有效用保证；与标准扰动分析不同，使用私有主成分分析过程避免了维度诅咒的影响。 |
| [^2] | [Mean-variance constrained priors have finite maximum Bayes risk in the normal location model.](http://arxiv.org/abs/2303.08653) | 该研究证明了在正态位置模型中，限制了均值方差的先验，在贝叶斯风险下具有有限的最大误差，结果适用于各种先验和方差。 |

# 详细

[^1]: 高维数据的差分隐私低维表示

    Differentially private low-dimensional representation of high-dimensional data. (arXiv:2305.17148v1 [cs.LG])

    [http://arxiv.org/abs/2305.17148](http://arxiv.org/abs/2305.17148)

    本文提出了一种在保护个人敏感信息的情况下，生成高效低维合成数据的算法，并在Wasserstein距离方面具有效用保证；与标准扰动分析不同，使用私有主成分分析过程避免了维度诅咒的影响。

    

    差分隐私合成数据提供了一种有效的机制，可以在保护个人敏感信息的同时进行数据分析。然而，当数据处于高维空间中时，合成数据的准确性会受到维度诅咒的影响。在本文中，我们提出了一种差分隐私算法，可以从高维数据集中高效地生成低维合成数据，并在Wasserstein距离方面具有效用保证。我们算法的一个关键步骤是使用具有近乎最优精度界限的私有主成分分析（PCA）过程，从而规避了维度诅咒的影响。与使用Davis-Kahan定理进行标准扰动分析不同，我们的私有PCA分析不需要假设样本协方差矩阵的谱间隙。

    Differentially private synthetic data provide a powerful mechanism to enable data analysis while protecting sensitive information about individuals. However, when the data lie in a high-dimensional space, the accuracy of the synthetic data suffers from the curse of dimensionality. In this paper, we propose a differentially private algorithm to generate low-dimensional synthetic data efficiently from a high-dimensional dataset with a utility guarantee with respect to the Wasserstein distance. A key step of our algorithm is a private principal component analysis (PCA) procedure with a near-optimal accuracy bound that circumvents the curse of dimensionality. Different from the standard perturbation analysis using the Davis-Kahan theorem, our analysis of private PCA works without assuming the spectral gap for the sample covariance matrix.
    
[^2]: 正态位置模型中的均值-方差受限先验有有限的最大贝叶斯风险

    Mean-variance constrained priors have finite maximum Bayes risk in the normal location model. (arXiv:2303.08653v1 [math.ST])

    [http://arxiv.org/abs/2303.08653](http://arxiv.org/abs/2303.08653)

    该研究证明了在正态位置模型中，限制了均值方差的先验，在贝叶斯风险下具有有限的最大误差，结果适用于各种先验和方差。

    

    考虑一个正态位置模型，其中 $X \mid \theta \sim N(\theta, \sigma^2)$，$\sigma^2$已知。假设 $\theta \sim G_0$，其中先验 $G_0$ 具有零均值和单位方差。令 $G_1$ 为可能存在误差的零均值和单位方差的先验。我们表明，在 $G_0, G_1, \sigma^2 > 0$ 范围内，贝叶斯风险下的后验均值的平方误差有界。

    Consider a normal location model $X \mid \theta \sim N(\theta, \sigma^2)$ with known $\sigma^2$. Suppose $\theta \sim G_0$, where the prior $G_0$ has zero mean and unit variance. Let $G_1$ be a possibly misspecified prior with zero mean and unit variance. We show that the squared error Bayes risk of the posterior mean under $G_1$ is bounded, uniformly over $G_0, G_1, \sigma^2 > 0$.
    

