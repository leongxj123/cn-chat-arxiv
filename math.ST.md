# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Differentially Private PCA and Estimation for Spiked Covariance Matrices.](http://arxiv.org/abs/2401.03820) | 本文研究了在尖峰协方差模型中的最优差分隐私主成分分析和协方差估计问题，并提出了高效的差分隐私估计器，并证明了它们的最小最大性。 |
| [^2] | [Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality.](http://arxiv.org/abs/2212.09900) | 本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。 |

# 详细

[^1]: 在带有尖峰协方差矩阵中的最优差分隐私主成分分析和估计

    Optimal Differentially Private PCA and Estimation for Spiked Covariance Matrices. (arXiv:2401.03820v1 [math.ST])

    [http://arxiv.org/abs/2401.03820](http://arxiv.org/abs/2401.03820)

    本文研究了在尖峰协方差模型中的最优差分隐私主成分分析和协方差估计问题，并提出了高效的差分隐私估计器，并证明了它们的最小最大性。

    

    在当代统计学中，估计协方差矩阵及其相关的主成分是一个基本问题。尽管已开发出具有良好性质的最优估计程序，但对隐私保护的增加需求给这个经典问题引入了新的复杂性。本文研究了在尖峰协方差模型中的最优差分隐私主成分分析（PCA）和协方差估计。我们精确地刻画了在该模型下特征值和特征向量的敏感性，并建立了估计主成分和协方差矩阵的最小最大收敛率。这些收敛率包括一般的Schatten范数，包括谱范数，Frobenius范数和核范数。我们引入了计算高效的差分隐私估计器，并证明它们的最小最大性，直到对数因子。另外，匹配的minimax最小最大率也得到了证明。

    Estimating a covariance matrix and its associated principal components is a fundamental problem in contemporary statistics. While optimal estimation procedures have been developed with well-understood properties, the increasing demand for privacy preservation introduces new complexities to this classical problem. In this paper, we study optimal differentially private Principal Component Analysis (PCA) and covariance estimation within the spiked covariance model.  We precisely characterize the sensitivity of eigenvalues and eigenvectors under this model and establish the minimax rates of convergence for estimating both the principal components and covariance matrix. These rates hold up to logarithmic factors and encompass general Schatten norms, including spectral norm, Frobenius norm, and nuclear norm as special cases.  We introduce computationally efficient differentially private estimators and prove their minimax optimality, up to logarithmic factors. Additionally, matching minimax l
    
[^2]: 无交叠策略学习：悲观和广义经验Bernstein不等式

    Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality. (arXiv:2212.09900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.09900](http://arxiv.org/abs/2212.09900)

    本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。

    

    本文研究了离线策略学习，旨在利用先前收集到的观测（来自于固定的或是适应演变的行为策略）来学习给定类别中的最优个性化决策规则。现有的策略学习方法依赖于一个统一交叠假设，即离线数据集中探索所有个性化特征的所有动作的倾向性下界。换句话说，这些方法的性能取决于离线数据集中最坏的倾向性。由于数据收集过程不受控制，在许多情况下，这种假设可能不太现实，特别是当允许行为策略随时间演变并且倾向性减弱时。为此，本文提出了一种新的算法，它优化策略价值的下限置信区间（LCBs）——而不是点估计。LCBs通过量化增强倒数倾向权重的估计不确定性来构建。

    This paper studies offline policy learning, which aims at utilizing observations collected a priori (from either fixed or adaptively evolving behavior policies) to learn the optimal individualized decision rule in a given class. Existing policy learning methods rely on a uniform overlap assumption, i.e., the propensities of exploring all actions for all individual characteristics are lower bounded in the offline dataset. In other words, the performance of these methods depends on the worst-case propensity in the offline dataset. As one has no control over the data collection process, this assumption can be unrealistic in many situations, especially when the behavior policies are allowed to evolve over time with diminishing propensities.  In this paper, we propose a new algorithm that optimizes lower confidence bounds (LCBs) -- instead of point estimates -- of the policy values. The LCBs are constructed by quantifying the estimation uncertainty of the augmented inverse propensity weight
    

