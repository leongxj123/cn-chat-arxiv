# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cutting Feedback in Misspecified Copula Models.](http://arxiv.org/abs/2310.03521) | 该论文介绍了一种在错配的Copula模型中限制反馈的剪切方法，并证明了在只有一个模块错配的情况下，适当的剪切后验提供了准确的不确定性量化。该方法在贝叶斯推断中具有重要的应用。 |
| [^2] | [Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models.](http://arxiv.org/abs/2307.09864) | 在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。 |
| [^3] | [A Penalized Poisson Likelihood Approach to High-Dimensional Semi-Parametric Inference for Doubly-Stochastic Point Processes.](http://arxiv.org/abs/2306.06756) | 本研究提出了一种对于双随机点过程的估计方法，该方法在进行协变量效应估计时非常高效，不需要强烈的限制性假设，且在理论和实践中均表现出了良好的信度保证和效能。 |

# 详细

[^1]: 在错配的Copula模型中限制反馈的剪切方法

    Cutting Feedback in Misspecified Copula Models. (arXiv:2310.03521v1 [stat.ME])

    [http://arxiv.org/abs/2310.03521](http://arxiv.org/abs/2310.03521)

    该论文介绍了一种在错配的Copula模型中限制反馈的剪切方法，并证明了在只有一个模块错配的情况下，适当的剪切后验提供了准确的不确定性量化。该方法在贝叶斯推断中具有重要的应用。

    

    在Copula模型中，边缘分布和Copula函数被分别指定。我们将它们视为模块化贝叶斯推断框架中的两个模块，并提出通过“剪切反馈”进行修改的贝叶斯推断方法。剪切反馈限制了后验推断中潜在错配模块的影响。我们考虑两种类型的剪切方法。第一种限制了错配Copula对边缘推断的影响，这是流行的边际推断（IFM）估计的贝叶斯类似方法。第二种通过使用秩似然定义剪切模型来限制错配边缘对Copula参数推断的影响。我们证明，如果只有一个模块错配，那么适当的剪切后验在另一个模块的参数的渐近不确定性量化方面是准确的。计算剪切后验很困难，我们提出了新的变分推断方法来解决这个问题。

    In copula models the marginal distributions and copula function are specified separately. We treat these as two modules in a modular Bayesian inference framework, and propose conducting modified Bayesian inference by ``cutting feedback''. Cutting feedback limits the influence of potentially misspecified modules in posterior inference. We consider two types of cuts. The first limits the influence of a misspecified copula on inference for the marginals, which is a Bayesian analogue of the popular Inference for Margins (IFM) estimator. The second limits the influence of misspecified marginals on inference for the copula parameters by using a rank likelihood to define the cut model. We establish that if only one of the modules is misspecified, then the appropriate cut posterior gives accurate uncertainty quantification asymptotically for the parameters in the other module. Computation of the cut posteriors is difficult, and new variational inference methods to do so are proposed. The effic
    
[^2]: 大型近似因子模型中主成分和准极大似然估计量的渐近等价性分析

    Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models. (arXiv:2307.09864v1 [econ.EM])

    [http://arxiv.org/abs/2307.09864](http://arxiv.org/abs/2307.09864)

    在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。

    

    我们证明在一个$n$维的稳定时间序列向量的近似因子模型中，通过主成分估计的因子载荷在$n\to\infty$时与准极大似然估计得到的载荷等价。这两种估计量在$n\to\infty$时也与如果观察到因子时的不可行最小二乘估计等价。我们还证明了准极大似然估计的渐近协方差矩阵的传统三明治形式与不可行最小二乘的简单渐近协方差矩阵等价。这提供了一种简单的方法来估计准极大似然估计的渐近置信区间，而不需要估计复杂的海森矩阵和费谢尔信息矩阵。所有结果均适用于假设异方差跨截面的一般情况。

    We prove that in an approximate factor model for an $n$-dimensional vector of stationary time series the factor loadings estimated via Principal Components are asymptotically equivalent, as $n\to\infty$, to those estimated by Quasi Maximum Likelihood. Both estimators are, in turn, also asymptotically equivalent, as $n\to\infty$, to the unfeasible Ordinary Least Squares estimator we would have if the factors were observed. We also show that the usual sandwich form of the asymptotic covariance matrix of the Quasi Maximum Likelihood estimator is asymptotically equivalent to the simpler asymptotic covariance matrix of the unfeasible Ordinary Least Squares. This provides a simple way to estimate asymptotic confidence intervals for the Quasi Maximum Likelihood estimator without the need of estimating the Hessian and Fisher information matrices whose expressions are very complex. All our results hold in the general case in which the idiosyncratic components are cross-sectionally heteroskedast
    
[^3]: 一种对于双随机点过程的高维半参数推理的惩罚泊松似然方法。

    A Penalized Poisson Likelihood Approach to High-Dimensional Semi-Parametric Inference for Doubly-Stochastic Point Processes. (arXiv:2306.06756v1 [stat.ME])

    [http://arxiv.org/abs/2306.06756](http://arxiv.org/abs/2306.06756)

    本研究提出了一种对于双随机点过程的估计方法，该方法在进行协变量效应估计时非常高效，不需要强烈的限制性假设，且在理论和实践中均表现出了良好的信度保证和效能。

    

    双随机点过程将空间域内事件的发生建模为在实现随机强度函数的条件下，不均匀泊松过程。它们是捕捉空间异质性和依赖性的灵活工具。然而，双随机空间模型的实现在计算上是有要求的，往往具有有限的理论保证和/或依赖于具有限制性假设。我们提出了一种惩罚回归方法，用于估计双随机点过程中的协变量效应，具有计算效率且不需要基础强度的参数形式或平稳性。我们证实了所提出估计器的一致性和渐近正态性，并开发了一个协方差估计器，导致保守的统计推断程序。模拟研究显示了我们的方法在数据生成机制的限制性较小的情况下的有效性，并且在西雅图犯罪事件的应用中证明了我们的方法在实践中的良好性能。

    Doubly-stochastic point processes model the occurrence of events over a spatial domain as an inhomogeneous Poisson process conditioned on the realization of a random intensity function. They are flexible tools for capturing spatial heterogeneity and dependence. However, implementations of doubly-stochastic spatial models are computationally demanding, often have limited theoretical guarantee, and/or rely on restrictive assumptions. We propose a penalized regression method for estimating covariate effects in doubly-stochastic point processes that is computationally efficient and does not require a parametric form or stationarity of the underlying intensity. We establish the consistency and asymptotic normality of the proposed estimator, and develop a covariance estimator that leads to a conservative statistical inference procedure. A simulation study shows the validity of our approach under less restrictive assumptions on the data generating mechanism, and an application to Seattle crim
    

