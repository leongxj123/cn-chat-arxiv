# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimating heterogeneous treatment effect from survival outcomes via (orthogonal) censoring unbiased learning.](http://arxiv.org/abs/2401.11263) | 该论文开发了一种适用于具有和没有竞争风险的生存结果的截尾无偏变换方法，可以估计异质治疗效应。这种方法可以应用于更多最先进的适用于被截尾结果的HTE学习方法，并提供了限制有限样本过度风险的方法。 |
| [^2] | [Optimal Differentially Private PCA and Estimation for Spiked Covariance Matrices.](http://arxiv.org/abs/2401.03820) | 本文研究了在尖峰协方差模型中的最优差分隐私主成分分析和协方差估计问题，并提出了高效的差分隐私估计器，并证明了它们的最小最大性。 |
| [^3] | [Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality.](http://arxiv.org/abs/2212.09900) | 本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。 |

# 详细

[^1]: 通过（正交）完全无偏的截尾学习来估计生存结果的异质治疗效应

    Estimating heterogeneous treatment effect from survival outcomes via (orthogonal) censoring unbiased learning. (arXiv:2401.11263v1 [stat.ME])

    [http://arxiv.org/abs/2401.11263](http://arxiv.org/abs/2401.11263)

    该论文开发了一种适用于具有和没有竞争风险的生存结果的截尾无偏变换方法，可以估计异质治疗效应。这种方法可以应用于更多最先进的适用于被截尾结果的HTE学习方法，并提供了限制有限样本过度风险的方法。

    

    从观察数据中估计异质治疗效应（HTE）的方法主要集中在连续或二元结果上，较少关注生存结果，几乎没有关注竞争风险情景。在这项工作中，我们开发了适用于具有和没有竞争风险的生存结果的截尾无偏变换（CUTs）。使用这些CUTs将时间到事件结果转换后，对连续结果的HTE学习方法的直接应用可以产生一致估计的异质累积发生率效应、总效应和可分离直接效应。我们的CUTs可以使用比以前更多的最先进的适用于被截尾结果的HTE学习方法，特别是在竞争风险情景下。我们提供了通用的无模型学习特定oracle不等式来限制有限样本的过度风险。oracle效率结果取决于一个oracle选择器和从所有步骤中估计的干扰函数。

    Methods for estimating heterogeneous treatment effects (HTE) from observational data have largely focused on continuous or binary outcomes, with less attention paid to survival outcomes and almost none to settings with competing risks. In this work, we develop censoring unbiased transformations (CUTs) for survival outcomes both with and without competing risks.After converting time-to-event outcomes using these CUTs, direct application of HTE learners for continuous outcomes yields consistent estimates of heterogeneous cumulative incidence effects, total effects, and separable direct effects. Our CUTs enable application of a much larger set of state of the art HTE learners for censored outcomes than had previously been available, especially in competing risks settings. We provide generic model-free learner-specific oracle inequalities bounding the finite-sample excess risk. The oracle efficiency results depend on the oracle selector and estimated nuisance functions from all steps invol
    
[^2]: 在带有尖峰协方差矩阵中的最优差分隐私主成分分析和估计

    Optimal Differentially Private PCA and Estimation for Spiked Covariance Matrices. (arXiv:2401.03820v1 [math.ST])

    [http://arxiv.org/abs/2401.03820](http://arxiv.org/abs/2401.03820)

    本文研究了在尖峰协方差模型中的最优差分隐私主成分分析和协方差估计问题，并提出了高效的差分隐私估计器，并证明了它们的最小最大性。

    

    在当代统计学中，估计协方差矩阵及其相关的主成分是一个基本问题。尽管已开发出具有良好性质的最优估计程序，但对隐私保护的增加需求给这个经典问题引入了新的复杂性。本文研究了在尖峰协方差模型中的最优差分隐私主成分分析（PCA）和协方差估计。我们精确地刻画了在该模型下特征值和特征向量的敏感性，并建立了估计主成分和协方差矩阵的最小最大收敛率。这些收敛率包括一般的Schatten范数，包括谱范数，Frobenius范数和核范数。我们引入了计算高效的差分隐私估计器，并证明它们的最小最大性，直到对数因子。另外，匹配的minimax最小最大率也得到了证明。

    Estimating a covariance matrix and its associated principal components is a fundamental problem in contemporary statistics. While optimal estimation procedures have been developed with well-understood properties, the increasing demand for privacy preservation introduces new complexities to this classical problem. In this paper, we study optimal differentially private Principal Component Analysis (PCA) and covariance estimation within the spiked covariance model.  We precisely characterize the sensitivity of eigenvalues and eigenvectors under this model and establish the minimax rates of convergence for estimating both the principal components and covariance matrix. These rates hold up to logarithmic factors and encompass general Schatten norms, including spectral norm, Frobenius norm, and nuclear norm as special cases.  We introduce computationally efficient differentially private estimators and prove their minimax optimality, up to logarithmic factors. Additionally, matching minimax l
    
[^3]: 无交叠策略学习：悲观和广义经验Bernstein不等式

    Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality. (arXiv:2212.09900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.09900](http://arxiv.org/abs/2212.09900)

    本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。

    

    本文研究了离线策略学习，旨在利用先前收集到的观测（来自于固定的或是适应演变的行为策略）来学习给定类别中的最优个性化决策规则。现有的策略学习方法依赖于一个统一交叠假设，即离线数据集中探索所有个性化特征的所有动作的倾向性下界。换句话说，这些方法的性能取决于离线数据集中最坏的倾向性。由于数据收集过程不受控制，在许多情况下，这种假设可能不太现实，特别是当允许行为策略随时间演变并且倾向性减弱时。为此，本文提出了一种新的算法，它优化策略价值的下限置信区间（LCBs）——而不是点估计。LCBs通过量化增强倒数倾向权重的估计不确定性来构建。

    This paper studies offline policy learning, which aims at utilizing observations collected a priori (from either fixed or adaptively evolving behavior policies) to learn the optimal individualized decision rule in a given class. Existing policy learning methods rely on a uniform overlap assumption, i.e., the propensities of exploring all actions for all individual characteristics are lower bounded in the offline dataset. In other words, the performance of these methods depends on the worst-case propensity in the offline dataset. As one has no control over the data collection process, this assumption can be unrealistic in many situations, especially when the behavior policies are allowed to evolve over time with diminishing propensities.  In this paper, we propose a new algorithm that optimizes lower confidence bounds (LCBs) -- instead of point estimates -- of the policy values. The LCBs are constructed by quantifying the estimation uncertainty of the augmented inverse propensity weight
    

