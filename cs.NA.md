# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric Estimation via Variance-Reduced Sketching.](http://arxiv.org/abs/2401.11646) | 本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。 |

# 详细

[^1]: 通过方差降低的草图进行非参数估计

    Nonparametric Estimation via Variance-Reduced Sketching. (arXiv:2401.11646v1 [stat.ML])

    [http://arxiv.org/abs/2401.11646](http://arxiv.org/abs/2401.11646)

    本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。

    

    非参数模型在各个科学和工程领域中备受关注。经典的核方法在低维情况下具有数值稳定性和统计可靠性，但在高维情况下由于维度灾难变得不够适用。在本文中，我们引入了一个名为Variance-Reduced Sketching（VRS）的新框架，专门用于在降低维度灾难的同时在高维度中估计密度函数和非参数回归函数。我们的框架将多变量函数概念化为无限大小的矩阵，并借鉴了数值线性代数文献中的一种新的草图技术来降低估计问题中的方差。我们通过一系列的模拟实验和真实数据应用展示了VRS的鲁棒性能。值得注意的是，在许多密度估计问题中，VRS相较于现有的神经网络估计器和经典的核方法表现出显著的改进。

    Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density 
    

