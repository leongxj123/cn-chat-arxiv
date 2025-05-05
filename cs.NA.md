# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric Estimation via Variance-Reduced Sketching.](http://arxiv.org/abs/2401.11646) | 本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。 |
| [^2] | [Generating synthetic data for neural operators.](http://arxiv.org/abs/2401.02398) | 该论文提出了一种生成神经算子的合成数据的新方法，为训练网络提供不需要数值求解PDE的数据。 |
| [^3] | [Learning the hub graphical Lasso model with the structured sparsity via an efficient algorithm.](http://arxiv.org/abs/2308.08852) | 通过双重交替方向乘子法 (ADMM) 和半平滑牛顿 (SSN) 基于增广对偶法 (ALM) 的方法，我们提出了一个高效算法来学习具有结构稀疏性的核心图形Lasso模型，该算法能够在大维度的任务中节省超过70\%的执行时间，并且具有较高的性能。 |

# 详细

[^1]: 通过方差降低的草图进行非参数估计

    Nonparametric Estimation via Variance-Reduced Sketching. (arXiv:2401.11646v1 [stat.ML])

    [http://arxiv.org/abs/2401.11646](http://arxiv.org/abs/2401.11646)

    本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。

    

    非参数模型在各个科学和工程领域中备受关注。经典的核方法在低维情况下具有数值稳定性和统计可靠性，但在高维情况下由于维度灾难变得不够适用。在本文中，我们引入了一个名为Variance-Reduced Sketching（VRS）的新框架，专门用于在降低维度灾难的同时在高维度中估计密度函数和非参数回归函数。我们的框架将多变量函数概念化为无限大小的矩阵，并借鉴了数值线性代数文献中的一种新的草图技术来降低估计问题中的方差。我们通过一系列的模拟实验和真实数据应用展示了VRS的鲁棒性能。值得注意的是，在许多密度估计问题中，VRS相较于现有的神经网络估计器和经典的核方法表现出显著的改进。

    Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density 
    
[^2]: 生成神经算子的合成数据

    Generating synthetic data for neural operators. (arXiv:2401.02398v1 [cs.LG])

    [http://arxiv.org/abs/2401.02398](http://arxiv.org/abs/2401.02398)

    该论文提出了一种生成神经算子的合成数据的新方法，为训练网络提供不需要数值求解PDE的数据。

    

    近期文献中的许多发展展示了深度学习在获取偏微分方程（PDEs）的数值解方面的潜力，这超出了当前数值求解器的能力。然而，数据驱动的神经算子都存在同样的问题：训练网络所需的数据依赖于传统的数值求解器，如有限差分或有限元等。本文提出了一种新方法，用于生成合成的函数训练数据，而无需数值求解PDE。我们的方法很简单：我们从已知解位于的经典理论解空间（例如$H_0^1(\Omega)$）中抽取大量独立同分布的“随机函数”$u_j$，然后将每个随机解方案代入方程并获得相应的右侧函数$f_j$，将$(f_j, u_j)_{j=1}^N$作为监督训练数据。

    Numerous developments in the recent literature show the promising potential of deep learning in obtaining numerical solutions to partial differential equations (PDEs) beyond the reach of current numerical solvers. However, data-driven neural operators all suffer from the same problem: the data needed to train a network depends on classical numerical solvers such as finite difference or finite element, among others. In this paper, we propose a new approach to generating synthetic functional training data that does not require solving a PDE numerically. The way we do this is simple: we draw a large number $N$ of independent and identically distributed `random functions' $u_j$ from the underlying solution space (e.g., $H_0^1(\Omega)$) in which we know the solution lies according to classical theory. We then plug each such random candidate solution into the equation and get a corresponding right-hand side function $f_j$ for the equation, and consider $(f_j, u_j)_{j=1}^N$ as supervised trai
    
[^3]: 通过高效算法学习具有结构稀疏性的核心图形Lasso模型

    Learning the hub graphical Lasso model with the structured sparsity via an efficient algorithm. (arXiv:2308.08852v1 [math.OC])

    [http://arxiv.org/abs/2308.08852](http://arxiv.org/abs/2308.08852)

    通过双重交替方向乘子法 (ADMM) 和半平滑牛顿 (SSN) 基于增广对偶法 (ALM) 的方法，我们提出了一个高效算法来学习具有结构稀疏性的核心图形Lasso模型，该算法能够在大维度的任务中节省超过70\%的执行时间，并且具有较高的性能。

    

    图形模型在从生物分析到推荐系统等众多任务中展现出了良好的性能。然而，具有核心节点的图形模型在数据维度较大时计算上存在困难。为了高效估计核心图形模型，我们提出了一个两阶段算法。所提出的算法首先通过双重交替方向乘子法 (ADMM) 生成一个良好的初始点，然后使用半平滑牛顿 (SSN) 基于增广对偶法 (ALM) 的方法进行热启动，以计算出能够在实际任务中精确到足够程度的解。广义雅可比矩阵的稀疏结构确保了该算法能够非常高效地获得一个良好的解。在合成数据和真实数据的全面实验中，该算法明显优于现有的最先进算法。特别是在某些高维任务中，它可以节省超过70\%的执行时间，同时仍然可以达到很好的性能。

    Graphical models have exhibited their performance in numerous tasks ranging from biological analysis to recommender systems. However, graphical models with hub nodes are computationally difficult to fit, particularly when the dimension of the data is large. To efficiently estimate the hub graphical models, we introduce a two-phase algorithm. The proposed algorithm first generates a good initial point via a dual alternating direction method of multipliers (ADMM), and then warm starts a semismooth Newton (SSN) based augmented Lagrangian method (ALM) to compute a solution that is accurate enough for practical tasks. The sparsity structure of the generalized Jacobian ensures that the algorithm can obtain a nice solution very efficiently. Comprehensive experiments on both synthetic data and real data show that it obviously outperforms the existing state-of-the-art algorithms. In particular, in some high dimensional tasks, it can save more than 70\% of the execution time, meanwhile still ach
    

