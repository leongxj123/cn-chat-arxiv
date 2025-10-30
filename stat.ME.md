# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression.](http://arxiv.org/abs/2310.13966) | 本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。 |
| [^2] | [Latent Factor Analysis in Short Panels.](http://arxiv.org/abs/2306.14004) | 本研究提出了短面板中潜在因子分析的推理工具，生成似然比统计量并得出了AUMPI特征，经实证应用发现短子期特有波动率呈上升趋势。 |

# 详细

[^1]: 基于核非参数回归的最优极小化传递学习

    Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression. (arXiv:2310.13966v1 [stat.ML])

    [http://arxiv.org/abs/2310.13966](http://arxiv.org/abs/2310.13966)

    本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。

    

    近年来，传递学习在机器学习社区中受到了很大关注。它能够利用相关研究的知识来提高目标研究的泛化性能，使其具有很高的吸引力。本文主要研究在再生核希尔伯特空间中的非参数回归的传递学习问题，目的是缩小实际效果与理论保证之间的差距。具体考虑了两种情况：已知可传递的来源和未知的情况。对于已知可传递的来源情况，我们提出了一个两步核估计器，仅使用核岭回归。对于未知的情况，我们开发了一种基于高效聚合算法的新方法，可以自动检测并减轻负面来源的影响。本文提供了所需估计器的统计性质，并建立了该方法的最优性结果。

    In recent years, transfer learning has garnered significant attention in the machine learning community. Its ability to leverage knowledge from related studies to improve generalization performance in a target study has made it highly appealing. This paper focuses on investigating the transfer learning problem within the context of nonparametric regression over a reproducing kernel Hilbert space. The aim is to bridge the gap between practical effectiveness and theoretical guarantees. We specifically consider two scenarios: one where the transferable sources are known and another where they are unknown. For the known transferable source case, we propose a two-step kernel-based estimator by solely using kernel ridge regression. For the unknown case, we develop a novel method based on an efficient aggregation algorithm, which can automatically detect and alleviate the effects of negative sources. This paper provides the statistical properties of the desired estimators and establishes the 
    
[^2]: 短面板中的潜在因子分析

    Latent Factor Analysis in Short Panels. (arXiv:2306.14004v1 [econ.EM])

    [http://arxiv.org/abs/2306.14004](http://arxiv.org/abs/2306.14004)

    本研究提出了短面板中潜在因子分析的推理工具，生成似然比统计量并得出了AUMPI特征，经实证应用发现短子期特有波动率呈上升趋势。

    

    我们开发了短面板中潜在因子分析的推理工具。 在大的横截面维度n和固定时间序列维度T的伪最大似然设置中，依赖于错误的对角线T×T协方差矩阵，而不强加球形或高斯性。 我们概述了潜在因子和误差协方差估计的渐近分布，以及基于似然比统计量的渐近一致最有力不变（AUMPI）测试的渐近分布，测试因子的数量。 我们从确保正态变量中正定二次形式的单调似然比属性的不等式中导出了AUMPI特征。 对美国一大批月度股票收益的实证应用基于所选因子数量，将短子期的牛市与熊市之后的日期系统和特有风险分开。 我们观察到，样本期间特有波动率呈上升趋势，而系统风险保持稳定。

    We develop inferential tools for latent factor analysis in short panels. The pseudo maximum likelihood setting under a large cross-sectional dimension $n$ and a fixed time series dimension $T$ relies on a diagonal $T \times T$ covariance matrix of the errors without imposing sphericity or Gaussianity. We outline the asymptotic distributions of the latent factor and error covariance estimates as well as of an asymptotically uniformly most powerful invariant (AUMPI) test based on the likelihood ratio statistic for tests of the number of factors. We derive the AUMPI characterization from inequalities ensuring the monotone likelihood ratio property for positive definite quadratic forms in normal variables. An empirical application to a large panel of monthly U.S. stock returns separates date after date systematic and idiosyncratic risks in short subperiods of bear vs. bull market based on the selected number of factors. We observe an uptrend in idiosyncratic volatility while the systematic
    

