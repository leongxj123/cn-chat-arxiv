# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric Estimation via Variance-Reduced Sketching.](http://arxiv.org/abs/2401.11646) | 本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。 |
| [^2] | [Model-based causal feature selection for general response types.](http://arxiv.org/abs/2309.12833) | 本研究基于模型提出了一种通用响应类型的因果特征选择方法，该方法利用不变性假设从异质环境的数据中输出一部分因果特征的子集，适用于一般的加性噪声模型和非参数设置，解决了非参数条件独立性测试低功率的问题。 |

# 详细

[^1]: 通过方差降低的草图进行非参数估计

    Nonparametric Estimation via Variance-Reduced Sketching. (arXiv:2401.11646v1 [stat.ML])

    [http://arxiv.org/abs/2401.11646](http://arxiv.org/abs/2401.11646)

    本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。

    

    非参数模型在各个科学和工程领域中备受关注。经典的核方法在低维情况下具有数值稳定性和统计可靠性，但在高维情况下由于维度灾难变得不够适用。在本文中，我们引入了一个名为Variance-Reduced Sketching（VRS）的新框架，专门用于在降低维度灾难的同时在高维度中估计密度函数和非参数回归函数。我们的框架将多变量函数概念化为无限大小的矩阵，并借鉴了数值线性代数文献中的一种新的草图技术来降低估计问题中的方差。我们通过一系列的模拟实验和真实数据应用展示了VRS的鲁棒性能。值得注意的是，在许多密度估计问题中，VRS相较于现有的神经网络估计器和经典的核方法表现出显著的改进。

    Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density 
    
[^2]: 基于模型的通用响应类型因果特征选择

    Model-based causal feature selection for general response types. (arXiv:2309.12833v1 [stat.ME])

    [http://arxiv.org/abs/2309.12833](http://arxiv.org/abs/2309.12833)

    本研究基于模型提出了一种通用响应类型的因果特征选择方法，该方法利用不变性假设从异质环境的数据中输出一部分因果特征的子集，适用于一般的加性噪声模型和非参数设置，解决了非参数条件独立性测试低功率的问题。

    

    从观测数据中发现因果关系是一项基本而具有挑战性的任务。在某些应用中，仅学习给定响应变量的因果特征可能已经足够，而不是学习整个潜在的因果结构。不变因果预测（ICP）是一种用于因果特征选择的方法，需要来自异质环境的数据。ICP假设从直接原因生成响应的机制在所有环境中都相同，并利用这种不变性输出一部分因果特征的子集。ICP的框架已经扩展到一般的加性噪声模型和非参数设置，使用条件独立性测试。然而，非参数条件独立性测试经常受到低功率（或较差的类型I错误控制）的困扰，并且上述参数模型不适用于响应不是在连续刻度上测量的应用情况，而是反映了分类信息的情况。

    Discovering causal relationships from observational data is a fundamental yet challenging task. In some applications, it may suffice to learn the causal features of a given response variable, instead of learning the entire underlying causal structure. Invariant causal prediction (ICP, Peters et al., 2016) is a method for causal feature selection which requires data from heterogeneous settings. ICP assumes that the mechanism for generating the response from its direct causes is the same in all settings and exploits this invariance to output a subset of the causal features. The framework of ICP has been extended to general additive noise models and to nonparametric settings using conditional independence testing. However, nonparametric conditional independence testing often suffers from low power (or poor type I error control) and the aforementioned parametric models are not suitable for applications in which the response is not measured on a continuous scale, but rather reflects categor
    

