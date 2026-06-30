# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Bayesian inference for the generalized linear mixed model](https://arxiv.org/abs/2403.03007) | 该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。 |
| [^2] | [Temporal-spatial model via Trend Filtering.](http://arxiv.org/abs/2308.16172) | 本研究通过趋势滤波方法对具有时空依赖性的数据进行了非参数回归函数的估计，研究了该方法在单变量和多变量情况下的应用，并验证了其极小化性。研究发现了以往未曾探索的独特相变现象，并通过仿真和实际数据应用验证了方法的优越性能。 |

# 详细

[^1]: 通用线性混合模型的可扩展贝叶斯推断

    Scalable Bayesian inference for the generalized linear mixed model

    [https://arxiv.org/abs/2403.03007](https://arxiv.org/abs/2403.03007)

    该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。

    

    通用线性混合模型（GLMM）是处理相关数据的一种流行统计方法，在包括生物医学数据等大数据常见的应用领域被广泛使用。本文的重点是针对GLMM的可扩展统计推断，我们将统计推断定义为：（i）对总体参数的估计以及（ii）在存在不确定性的情况下评估科学假设。人工智能（AI）学习算法擅长可扩展的统计估计，但很少包括不确定性量化。相比之下，贝叶斯推断提供完整的统计推断，因为不确定性量化自动来自后验分布。不幸的是，包括马尔可夫链蒙特卡洛（MCMC）在内的贝叶斯推断算法在大数据环境中变得难以计算。在本文中，我们介绍了一个统计推断算法

    arXiv:2403.03007v1 Announce Type: cross  Abstract: The generalized linear mixed model (GLMM) is a popular statistical approach for handling correlated data, and is used extensively in applications areas where big data is common, including biomedical data settings. The focus of this paper is scalable statistical inference for the GLMM, where we define statistical inference as: (i) estimation of population parameters, and (ii) evaluation of scientific hypotheses in the presence of uncertainty. Artificial intelligence (AI) learning algorithms excel at scalable statistical estimation, but rarely include uncertainty quantification. In contrast, Bayesian inference provides full statistical inference, since uncertainty quantification results automatically from the posterior distribution. Unfortunately, Bayesian inference algorithms, including Markov Chain Monte Carlo (MCMC), become computationally intractable in big data settings. In this paper, we introduce a statistical inference algorithm 
    
[^2]: 通过趋势滤波进行时空模型建模

    Temporal-spatial model via Trend Filtering. (arXiv:2308.16172v1 [stat.ME])

    [http://arxiv.org/abs/2308.16172](http://arxiv.org/abs/2308.16172)

    本研究通过趋势滤波方法对具有时空依赖性的数据进行了非参数回归函数的估计，研究了该方法在单变量和多变量情况下的应用，并验证了其极小化性。研究发现了以往未曾探索的独特相变现象，并通过仿真和实际数据应用验证了方法的优越性能。

    

    本研究侧重于对具有同时时间和空间依赖性的数据进行非参数回归函数的估计。在这种情况下，我们研究了趋势滤波，这是一种非参数估计方法，由Mammen和Rudin提出。在单变量设置中，我们考虑的信号假设具有有界总变异度的k次弱导数，允许一定程度的平滑性。在多变量情况下，我们研究了Padilla等人的K最近邻融合套索估计器，采用适用于具有有界变异度且符合分段利普希茨连续性准则的信号的ADMM算法。通过与下界对齐，我们验证了我们估计器的极小化性。通过分析，我们发现了以往趋势滤波研究中未曾探索过的独特相变现象。仿真研究和实际数据应用都突出了我们方法的出色性能。

    This research focuses on the estimation of a non-parametric regression function designed for data with simultaneous time and space dependencies. In such a context, we study the Trend Filtering, a nonparametric estimator introduced by \cite{mammen1997locally} and \cite{rudin1992nonlinear}. For univariate settings, the signals we consider are assumed to have a kth weak derivative with bounded total variation, allowing for a general degree of smoothness. In the multivariate scenario, we study a $K$-Nearest Neighbor fused lasso estimator as in \cite{padilla2018adaptive}, employing an ADMM algorithm, suitable for signals with bounded variation that adhere to a piecewise Lipschitz continuity criterion. By aligning with lower bounds, the minimax optimality of our estimators is validated. A unique phase transition phenomenon, previously uncharted in Trend Filtering studies, emerges through our analysis. Both Simulation studies and real data applications underscore the superior performance of o
    

