# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Functional Data Analysis for Stochastic Evolution Equations in Infinite Dimensions.](http://arxiv.org/abs/2401.16286) | 本文提出了一种在无限维度中解决随机演化方程的稳健函数数据分析方法，通过考虑横截面和时间结构的相互作用，有效地测量了协变量的变化，具有可辨识性和收敛速度，并能揭示与协变量的缩放极限一致。 |
| [^2] | [Sequential Gibbs Posteriors with Applications to Principal Component Analysis.](http://arxiv.org/abs/2310.12882) | 提出了一种新的序列扩展的Gibbs先验方法，解决了传统方法中的不确定性问题，并获得了关于伯恩斯坦-冯·密斯定理在流形上的新结果。 |
| [^3] | [Optimal Shrinkage Estimation of Fixed Effects in Linear Panel Data Models.](http://arxiv.org/abs/2308.12485) | 本文提出了一种在线性面板数据模型中估计固定效应的最优缩小估计方法，该方法不需要分布假设，并能够充分地利用序列相关性和时间变化。同时，还提供了一种预测未来固定效应的方法。 |

# 详细

[^1]: 无限维度中随机演化方程的稳健函数数据分析

    Robust Functional Data Analysis for Stochastic Evolution Equations in Infinite Dimensions. (arXiv:2401.16286v1 [stat.ME])

    [http://arxiv.org/abs/2401.16286](http://arxiv.org/abs/2401.16286)

    本文提出了一种在无限维度中解决随机演化方程的稳健函数数据分析方法，通过考虑横截面和时间结构的相互作用，有效地测量了协变量的变化，具有可辨识性和收敛速度，并能揭示与协变量的缩放极限一致。

    

    本文针对Hilbert空间中的随机演化方程，使用函数数据分析技术来稳健地测量协变量的变化。对于这样的方程，基于横截面协方差的标准技术经常无法识别出统计上相关的随机驱动因素和检测异常值，因为它们忽略了横截面和时间结构之间的相互作用。因此，我们开发了一种估计理论，用于估计方程的潜在随机驱动因素的连续二次协变量，而不是可观察解过程的静态协方差。我们在弱条件下得到了可辨识性的结果，建立了基于装满渐近性的收敛速度和中心极限定理，并提供了长时间渐近性的估计结果，以估计潜在驱动因素的静态协变量。应用于利率结构数据，我们的方法揭示了与协变量的缩放极限的根本一致性。

    This article addresses the robust measurement of covariations in the context of solutions to stochastic evolution equations in Hilbert spaces using functional data analysis. For such equations, standard techniques for functional data based on cross-sectional covariances are often inadequate for identifying statistically relevant random drivers and detecting outliers since they overlook the interplay between cross-sectional and temporal structures. Therefore, we develop an estimation theory for the continuous quadratic covariation of the latent random driver of the equation instead of a static covariance of the observable solution process. We derive identifiability results under weak conditions, establish rates of convergence and a central limit theorem based on infill asymptotics, and provide long-time asymptotics for estimation of a static covariation of the latent driver. Applied to term structure data, our approach uncovers a fundamental alignment with scaling limits of covariations
    
[^2]: 带有序列Gibbs先验的贡献于主成分分析的应用

    Sequential Gibbs Posteriors with Applications to Principal Component Analysis. (arXiv:2310.12882v1 [stat.ME])

    [http://arxiv.org/abs/2310.12882](http://arxiv.org/abs/2310.12882)

    提出了一种新的序列扩展的Gibbs先验方法，解决了传统方法中的不确定性问题，并获得了关于伯恩斯坦-冯·密斯定理在流形上的新结果。

    

    Gibbs先验与先验分布乘以指数损失函数成比例，其中关键调整参数在损失与先验中权重信息，并提供控制后验不确定性的能力。Gibbs先验为无似然贝叶斯推理提供了一个有原则的框架，但在许多情况下，使用单一调整参数会导致较差的不确定性量化。我们提出了一种序列扩展的Gibbs先验来解决这个问题。我们证明了所提出的序列后验展示了集中性和一个伯恩斯坦-冯·密斯定理，该定理在欧几里得空间和流形上易于验证的条件下成立。作为副产品，我们获得了传统基于似然的贝叶斯后验在流形上的第一个伯恩斯坦-冯·密斯定理。所有方法都有示例说明。

    Gibbs posteriors are proportional to a prior distribution multiplied by an exponentiated loss function, with a key tuning parameter weighting information in the loss relative to the prior and providing a control of posterior uncertainty. Gibbs posteriors provide a principled framework for likelihood-free Bayesian inference, but in many situations, including a single tuning parameter inevitably leads to poor uncertainty quantification. In particular, regardless of the value of the parameter, credible regions have far from the nominal frequentist coverage even in large samples. We propose a sequential extension to Gibbs posteriors to address this problem. We prove the proposed sequential posterior exhibits concentration and a Bernstein-von Mises theorem, which holds under easy to verify conditions in Euclidean space and on manifolds. As a byproduct, we obtain the first Bernstein-von Mises theorem for traditional likelihood-based Bayesian posteriors on manifolds. All methods are illustrat
    
[^3]: 线性面板数据模型中固定效应最优缩小估计

    Optimal Shrinkage Estimation of Fixed Effects in Linear Panel Data Models. (arXiv:2308.12485v1 [econ.EM])

    [http://arxiv.org/abs/2308.12485](http://arxiv.org/abs/2308.12485)

    本文提出了一种在线性面板数据模型中估计固定效应的最优缩小估计方法，该方法不需要分布假设，并能够充分地利用序列相关性和时间变化。同时，还提供了一种预测未来固定效应的方法。

    

    缩小估计方法经常被用于估计固定效应，以减少最小二乘估计的噪声。然而，广泛使用的缩小估计仅在强分布假设下才能保证降低噪声。本文开发了一种估计固定效应的估计器，在缩小估计器类别中获得了最佳的均方误差。该类别包括传统的缩小估计器，且最优性不需要分布假设。该估计器具有直观的形式，并且易于实现。此外，固定效应允许随时间变化，并且可以具有序列相关性，而缩小方法在这种情况下可以最优地结合底层相关结构。在这样的背景下，还提供了一种预测未来一个时期固定效应的方法。

    Shrinkage methods are frequently used to estimate fixed effects to reduce the noisiness of the least square estimators. However, widely used shrinkage estimators guarantee such noise reduction only under strong distributional assumptions. I develop an estimator for the fixed effects that obtains the best possible mean squared error within a class of shrinkage estimators. This class includes conventional shrinkage estimators and the optimality does not require distributional assumptions. The estimator has an intuitive form and is easy to implement. Moreover, the fixed effects are allowed to vary with time and to be serially correlated, and the shrinkage optimally incorporates the underlying correlation structure in this case. In such a context, I also provide a method to forecast fixed effects one period ahead.
    

