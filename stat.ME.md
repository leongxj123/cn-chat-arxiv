# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal testing in a class of nonregular models](https://arxiv.org/abs/2403.16413) | 本文研究了一类非常规模型中的最优假设检验，提出了基于似然比过程的渐近一致最强测试方法，通过随机化、调节常数和用户指定备择假设值等方式实现渐近最优性。 |
| [^2] | [Quick Adaptive Ternary Segmentation: An Efficient Decoding Procedure For Hidden Markov Models.](http://arxiv.org/abs/2305.18578) | 提出了一种名为QATS的新方法，用于高效解码隐藏马尔可夫模型序列。它的计算复杂性为多对数和立方，特别适用于具有相对较少状态的大型HMM。 |
| [^3] | [Change-Point Testing for Risk Measures in Time Series.](http://arxiv.org/abs/1809.02303) | 我们提出了一种变点检验方法，可以在时间序列的尾部检测到一般多重结构变化，并避免了标准误差估计的问题。实证研究表明我们的方法可以在金融时间序列中检测和量化市场不稳定性。 |

# 详细

[^1]: 一类非常规模型中的最优检验

    Optimal testing in a class of nonregular models

    [https://arxiv.org/abs/2403.16413](https://arxiv.org/abs/2403.16413)

    本文研究了一类非常规模型中的最优假设检验，提出了基于似然比过程的渐近一致最强测试方法，通过随机化、调节常数和用户指定备择假设值等方式实现渐近最优性。

    

    本文研究了参数依赖支持的非常规统计模型的最优假设检验。我们考虑了单侧和双侧假设检验，并基于似然比过程发展了渐近一致最强的检验。所提出的单侧检验涉及随机化以实现渐近尺寸控制，一些调节常数以避免在极限似然比过程中的不连续性，并一个用户指定的备择假设值以达到渐近最优性。我们的双侧检验在不施加进一步的限制（如无偏性）的情况下变为渐近一致最强。模拟结果展示了所提出检验的理想功效性质。

    arXiv:2403.16413v1 Announce Type: cross  Abstract: This paper studies optimal hypothesis testing for nonregular statistical models with parameter-dependent support. We consider both one-sided and two-sided hypothesis testing and develop asymptotically uniformly most powerful tests based on the likelihood ratio process. The proposed one-sided test involves randomization to achieve asymptotic size control, some tuning constant to avoid discontinuities in the limiting likelihood ratio process, and a user-specified alternative hypothetical value to achieve the asymptotic optimality. Our two-sided test becomes asymptotically uniformly most powerful without imposing further restrictions such as unbiasedness. Simulation results illustrate desirable power properties of the proposed tests.
    
[^2]: 快速自适应三元分割：隐马尔可夫模型的有效解码程序。

    Quick Adaptive Ternary Segmentation: An Efficient Decoding Procedure For Hidden Markov Models. (arXiv:2305.18578v1 [stat.ME])

    [http://arxiv.org/abs/2305.18578](http://arxiv.org/abs/2305.18578)

    提出了一种名为QATS的新方法，用于高效解码隐藏马尔可夫模型序列。它的计算复杂性为多对数和立方，特别适用于具有相对较少状态的大型HMM。

    

    隐马尔可夫模型（HMM）以不可观察的（隐藏的）马尔可夫链和可观测的过程为特征，后者是隐藏链的噪声版本。从嘈杂的观测中解码原始信号（即隐藏链）是几乎所有基于HMM的数据分析的主要目标。现有的解码算法，如维特比算法，在观测序列长度最多线性的情况下具有计算复杂度，并且在马尔可夫链状态空间的大小中具有次二次计算复杂度。我们提出了快速自适应三元分割（QATS），这是一种分而治之的过程，可在序列长度的多对数计算复杂度和马尔可夫链状态空间的三次计算复杂度下解码隐藏的序列，因此特别适用于具有相对较少状态的大规模HMM。该程序还建议一种有效的数据存储方式，即特定的累积总和。实质上，估计的状态序列按顺序最大化局部似然。

    Hidden Markov models (HMMs) are characterized by an unobservable (hidden) Markov chain and an observable process, which is a noisy version of the hidden chain. Decoding the original signal (i.e., hidden chain) from the noisy observations is one of the main goals in nearly all HMM based data analyses. Existing decoding algorithms such as the Viterbi algorithm have computational complexity at best linear in the length of the observed sequence, and sub-quadratic in the size of the state space of the Markov chain. We present Quick Adaptive Ternary Segmentation (QATS), a divide-and-conquer procedure which decodes the hidden sequence in polylogarithmic computational complexity in the length of the sequence, and cubic in the size of the state space, hence particularly suited for large scale HMMs with relatively few states. The procedure also suggests an effective way of data storage as specific cumulative sums. In essence, the estimated sequence of states sequentially maximizes local likeliho
    
[^3]: 时间序列中风险测度的变点检验

    Change-Point Testing for Risk Measures in Time Series. (arXiv:1809.02303v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1809.02303](http://arxiv.org/abs/1809.02303)

    我们提出了一种变点检验方法，可以在时间序列的尾部检测到一般多重结构变化，并避免了标准误差估计的问题。实证研究表明我们的方法可以在金融时间序列中检测和量化市场不稳定性。

    

    我们提出了一种新的方法，用于对弱相关时间序列中预期损失和相关风险测度的非参数估计进行变点检验。在一般假设下，我们可以检测到时间序列边缘分布尾部的一般多重结构变化。自归一化使我们能够避免标准误差估计的问题。我们的方法的理论基础是在弱假设下发展起来的函数中心极限定理。对S&P 500和美国国债回报的实证研究说明了我们的方法在通过金融时间序列的尾部检测和量化市场不稳定性方面的实际应用。

    We propose novel methods for change-point testing for nonparametric estimators of expected shortfall and related risk measures in weakly dependent time series. We can detect general multiple structural changes in the tails of marginal distributions of time series under general assumptions. Self-normalization allows us to avoid the issues of standard error estimation. The theoretical foundations for our methods are functional central limit theorems, which we develop under weak assumptions. An empirical study of S&P 500 and US Treasury bond returns illustrates the practical use of our methods in detecting and quantifying market instability via the tails of financial time series.
    

