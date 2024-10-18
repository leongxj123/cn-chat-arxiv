# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Iterative Estimation of Nonparametric Regressions with Continuous Endogenous Variables and Discrete Instruments](https://arxiv.org/abs/1905.07812) | 提出了一种简单的迭代程序来估计具有连续内生变量和离散工具的非参数回归模型，并展示了一些渐近性质。 |
| [^2] | [Concentration for high-dimensional linear processes with dependent innovations.](http://arxiv.org/abs/2307.12395) | 本论文提出了一种针对高维线性过程的具有相关创新的集中度不等式，并利用该不等式获得了线性过程滞后自协方差矩阵最大分量范数的集中度界限。这些结果在估计高维向量自回归过程、时间序列引导和长期协方差矩阵估计中具有重要应用价值。 |
| [^3] | [FAStEN: an efficient adaptive method for feature selection and estimation in high-dimensional functional regressions.](http://arxiv.org/abs/2303.14801) | 提出了一种新的自适应方法FAStEN，用于在高维函数回归问题中执行特征选择和参数估计，通过利用函数主成分和对偶增广Lagrangian问题的稀疏性质，具有显著的计算效率和选择准确性。 |

# 详细

[^1]: 连续内生变量和离散工具的非参数回归的迭代估计

    Iterative Estimation of Nonparametric Regressions with Continuous Endogenous Variables and Discrete Instruments

    [https://arxiv.org/abs/1905.07812](https://arxiv.org/abs/1905.07812)

    提出了一种简单的迭代程序来估计具有连续内生变量和离散工具的非参数回归模型，并展示了一些渐近性质。

    

    我们考虑了一个具有连续内生独立变量的非参数回归模型，当只有与误差项独立的离散工具可用时。虽然这个框架在应用研究中非常相关，但其实现很麻烦，因为回归函数成为了非线性积分方程的解。我们提出了一个简单的迭代过程来估计这样的模型，并展示了一些其渐近性质。在一个模拟实验中，我们讨论了在工具变量为二进制时其实现细节。我们总结了一个实证应用，其中我们研究了美国几个县的房价对污染的影响。

    arXiv:1905.07812v2 Announce Type: replace  Abstract: We consider a nonparametric regression model with continuous endogenous independent variables when only discrete instruments are available that are independent of the error term. While this framework is very relevant for applied research, its implementation is cumbersome, as the regression function becomes the solution to a nonlinear integral equation. We propose a simple iterative procedure to estimate such models and showcase some of its asymptotic properties. In a simulation experiment, we discuss the details of its implementation in the case when the instrumental variable is binary. We conclude with an empirical application in which we examine the effect of pollution on house prices in a short panel of U.S. counties.
    
[^2]: 高维线性过程中具有相关创新的集中度

    Concentration for high-dimensional linear processes with dependent innovations. (arXiv:2307.12395v1 [math.ST])

    [http://arxiv.org/abs/2307.12395](http://arxiv.org/abs/2307.12395)

    本论文提出了一种针对高维线性过程的具有相关创新的集中度不等式，并利用该不等式获得了线性过程滞后自协方差矩阵最大分量范数的集中度界限。这些结果在估计高维向量自回归过程、时间序列引导和长期协方差矩阵估计中具有重要应用价值。

    

    我们针对具有子韦布尔尾的混合序列上的线性过程的$l_\infty$范数开发了集中不等式。这些不等式利用了Beveridge-Nelson分解，将问题简化为向量混合序列或其加权和的上确界范数的集中度。这个不等式用于得到线性过程的滞后$h$自协方差矩阵的最大分量范数的集中度界限。这些结果对于使用$l_1$正则化估计的高维向量自回归过程的估计界限、时间序列的高维高斯引导、以及长期协方差矩阵估计非常有用。

    We develop concentration inequalities for the $l_\infty$ norm of a vector linear processes on mixingale sequences with sub-Weibull tails. These inequalities make use of the Beveridge-Nelson decomposition, which reduces the problem to concentration for sup-norm of a vector-mixingale or its weighted sum. This inequality is used to obtain a concentration bound for the maximum entrywise norm of the lag-$h$ autocovariance matrices of linear processes. These results are useful for estimation bounds for high-dimensional vector-autoregressive processes estimated using $l_1$ regularisation, high-dimensional Gaussian bootstrap for time series, and long-run covariance matrix estimation.
    
[^3]: 高维函数回归中特征选择和估计的一种高效自适应方法--FAStEN

    FAStEN: an efficient adaptive method for feature selection and estimation in high-dimensional functional regressions. (arXiv:2303.14801v1 [stat.ME])

    [http://arxiv.org/abs/2303.14801](http://arxiv.org/abs/2303.14801)

    提出了一种新的自适应方法FAStEN，用于在高维函数回归问题中执行特征选择和参数估计，通过利用函数主成分和对偶增广Lagrangian问题的稀疏性质，具有显著的计算效率和选择准确性。

    

    函数回归分析是许多当代科学应用的已建立工具。涉及大规模和复杂数据集的回归问题是普遍存在的，特征选择对于避免过度拟合和实现准确预测至关重要。我们提出了一种新的、灵活的、超高效的方法，用于在稀疏高维函数回归问题中执行特征选择，并展示了如何将其扩展到标量对函数框架中。我们的方法将函数数据、优化和机器学习技术相结合，以同时执行特征选择和参数估计。我们利用函数主成分的特性以及对偶增广Lagrangian问题的稀疏性质，显著降低了计算成本，并引入了自适应方案来提高选择准确性。通过广泛的模拟研究，我们将我们的方法与最佳现有竞争对手进行了基准测试，并证明了我们的方法的高效性。

    Functional regression analysis is an established tool for many contemporary scientific applications. Regression problems involving large and complex data sets are ubiquitous, and feature selection is crucial for avoiding overfitting and achieving accurate predictions. We propose a new, flexible, and ultra-efficient approach to perform feature selection in a sparse high dimensional function-on-function regression problem, and we show how to extend it to the scalar-on-function framework. Our method combines functional data, optimization, and machine learning techniques to perform feature selection and parameter estimation simultaneously. We exploit the properties of Functional Principal Components, and the sparsity inherent to the Dual Augmented Lagrangian problem to significantly reduce computational cost, and we introduce an adaptive scheme to improve selection accuracy. Through an extensive simulation study, we benchmark our approach to the best existing competitors and demonstrate a 
    

