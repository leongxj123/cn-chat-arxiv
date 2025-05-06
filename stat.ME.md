# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^2] | [Bayesian Analysis for Over-parameterized Linear Model without Sparsity.](http://arxiv.org/abs/2305.15754) | 本文提出了一种基于数据的特征向量的先验方法，用于处理非稀疏超参数线性模型。从导出的后验分布收缩率和开发的截断高斯近似两个方面来证明了该方法的有效性，可以解决之前的先验稀疏性限制。 |

# 详细

[^1]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^2]: 非稀疏超参数线性模型的贝叶斯分析

    Bayesian Analysis for Over-parameterized Linear Model without Sparsity. (arXiv:2305.15754v1 [math.ST])

    [http://arxiv.org/abs/2305.15754](http://arxiv.org/abs/2305.15754)

    本文提出了一种基于数据的特征向量的先验方法，用于处理非稀疏超参数线性模型。从导出的后验分布收缩率和开发的截断高斯近似两个方面来证明了该方法的有效性，可以解决之前的先验稀疏性限制。

    

    在高维贝叶斯统计学中，发展了许多方法，包括许多先验分布，它们导致估计参数的稀疏性。然而，这种先验在处理数据的谱特征向量结构方面有局限性，因此不适用于分析最近发展的不假设稀疏性的高维线性模型。本文介绍了一种贝叶斯方法，它使用一个依赖于数据协方差矩阵的特征向量的先验，但不会引起参数的稀疏性。我们还提供了导出的后验分布的收缩率，并开发了后验分布的截断高斯近似。前者证明了后验估计的效率，而后者则使用Bernstein-von Mises类型方法来量化参数不确定性。这些结果表明，任何能够处理谱特征向量的贝叶斯方法，都可以用于非稀疏超参数线性模型分析，从而解决了先前的限制。

    In high-dimensional Bayesian statistics, several methods have been developed, including many prior distributions that lead to the sparsity of estimated parameters. However, such priors have limitations in handling the spectral eigenvector structure of data, and as a result, they are ill-suited for analyzing over-parameterized models (high-dimensional linear models that do not assume sparsity) that have been developed in recent years. This paper introduces a Bayesian approach that uses a prior dependent on the eigenvectors of data covariance matrices, but does not induce the sparsity of parameters. We also provide contraction rates of derived posterior distributions and develop a truncated Gaussian approximation of the posterior distribution. The former demonstrates the efficiency of posterior estimation, while the latter enables quantification of parameter uncertainty using a Bernstein-von Mises-type approach. These results indicate that any Bayesian method that can handle the spectrum
    

