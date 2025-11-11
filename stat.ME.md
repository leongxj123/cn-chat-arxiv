# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection](https://arxiv.org/abs/2402.02196) | 我们提出了一种新颖的并行大规模排序和选择问题的聚类及征服方法，通过利用相关信息进行聚类以提高样本效率，在大规模AI应用中表现优异。 |
| [^2] | [Simple Estimation of Semiparametric Models with Measurement Errors.](http://arxiv.org/abs/2306.14311) | 本文提出了一种解决广义矩量方法（GMM）框架下变量误差（EIV）问题的方法，对于任何初始矩条件，该方法提供了纠正后对EIV具有鲁棒性的矩条件集，这使得GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论，对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。 |

# 详细

[^1]: 并行大规模排序和选择问题的样本高效聚类及征服方法

    Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection

    [https://arxiv.org/abs/2402.02196](https://arxiv.org/abs/2402.02196)

    我们提出了一种新颖的并行大规模排序和选择问题的聚类及征服方法，通过利用相关信息进行聚类以提高样本效率，在大规模AI应用中表现优异。

    

    我们提出了一种新颖的"聚类和征服"方法，用于解决并行大规模排序和选择问题，通过利用相关信息进行聚类，以打破样本效率的瓶颈。在并行计算环境中，基于相关性的聚类可以实现O(p)的样本复杂度减少速度，这是理论上可达到的最佳减少速度。我们提出的框架是通用的，在固定预算和固定精度的范式下，可以无缝集成各种常见的排序和选择方法。它可以在无需高精确度相关估计和精确聚类的情况下实现改进。在大规模人工智能应用中，如神经结构搜索，我们的无筛选版本的方法惊人地超过了完全顺序化的基准，表现出更高的样本效率。这表明利用有价值的结构信息，如相关性，是绕过传统方法的一条可行路径。

    We propose novel "clustering and conquer" procedures for the parallel large-scale ranking and selection (R&S) problem, which leverage correlation information for clustering to break the bottleneck of sample efficiency. In parallel computing environments, correlation-based clustering can achieve an $\mathcal{O}(p)$ sample complexity reduction rate, which is the optimal reduction rate theoretically attainable. Our proposed framework is versatile, allowing for seamless integration of various prevalent R&S methods under both fixed-budget and fixed-precision paradigms. It can achieve improvements without the necessity of highly accurate correlation estimation and precise clustering. In large-scale AI applications such as neural architecture search, a screening-free version of our procedure surprisingly surpasses fully-sequential benchmarks in terms of sample efficiency. This suggests that leveraging valuable structural information, such as correlation, is a viable path to bypassing the trad
    
[^2]: 测量误差中半参数模型的简单估计

    Simple Estimation of Semiparametric Models with Measurement Errors. (arXiv:2306.14311v1 [econ.EM])

    [http://arxiv.org/abs/2306.14311](http://arxiv.org/abs/2306.14311)

    本文提出了一种解决广义矩量方法（GMM）框架下变量误差（EIV）问题的方法，对于任何初始矩条件，该方法提供了纠正后对EIV具有鲁棒性的矩条件集，这使得GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论，对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。

    

    我们在广义矩量方法（GMM）框架下开发了一种解决变量误差（EIV）问题的实用方法。我们关注的是EIV的可变性是测量误差变量的一小部分的情况，这在实证应用中很常见。对于任何初始矩条件，我们的方法提供了纠正后对EIV具有鲁棒性的矩条件集。我们表明，基于这些矩的GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论。即使EIV很大，朴素估计量（忽略EIV问题）可能严重偏误并且置信区间的覆盖率为0％，我们的方法也能处理。我们的方法不涉及非参数估计，这对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。

    We develop a practical way of addressing the Errors-In-Variables (EIV) problem in the Generalized Method of Moments (GMM) framework. We focus on the settings in which the variability of the EIV is a fraction of that of the mismeasured variables, which is typical for empirical applications. For any initial set of moment conditions our approach provides a corrected set of moment conditions that are robust to the EIV. We show that the GMM estimator based on these moments is root-n-consistent, with the standard tests and confidence intervals providing valid inference. This is true even when the EIV are so large that naive estimators (that ignore the EIV problem) may be heavily biased with the confidence intervals having 0% coverage. Our approach involves no nonparametric estimation, which is particularly important for applications with multiple covariates, and settings with multivariate, serially correlated, or non-classical EIV.
    

