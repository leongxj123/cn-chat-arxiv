# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Coefficient Shape Alignment in Multivariate Functional Regression.](http://arxiv.org/abs/2312.01925) | 该论文提出了一种新的分组多元函数回归模型，其中采用了一种新的正则化方法来解决不同函数协变量的潜在同质性问题。 |
| [^2] | [Transport map unadjusted Langevin algorithms: learning and discretizing perturbed samplers.](http://arxiv.org/abs/2302.07227) | 本研究提出了交通图未调整的 Langevin 算法 (ULA) 和 Riemann 流形 Langevin 动力学 (RMLD)，通过应用交通图可以加速 Langevin 动力学的收敛，并提供了学习度量和扰动的新思路。 |

# 详细

[^1]: 在多元函数回归中的系数形状对齐

    Coefficient Shape Alignment in Multivariate Functional Regression. (arXiv:2312.01925v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2312.01925](http://arxiv.org/abs/2312.01925)

    该论文提出了一种新的分组多元函数回归模型，其中采用了一种新的正则化方法来解决不同函数协变量的潜在同质性问题。

    

    在多元函数数据分析中，不同的函数协变量可能具有同质性。隐藏的同质性结构对于不同协变量的连接或关联具有信息价值。具有明显同质性的协变量可以在同一群组中进行联合分析，从而产生一种简化建模多元函数数据的方法。本文提出了一种新颖的分组多元函数回归模型，采用称为“系数形状对齐”的新正则化方法来解决不同函数协变量的潜在同质性问题。建模过程包括两个主要步骤：首先，使用新的正则化方法检测未知分组结构，将协变量聚合到不相交的群组中；然后，基于检测到的分组结构建立分组多元函数回归模型。在这个新的分组模型中，同一同质群组中的系数函数应对齐。

    In multivariate functional data analysis, different functional covariates can be homogeneous. The hidden homogeneity structure is informative about the connectivity or association of different covariates. The covariates with pronounced homogeneity can be analyzed jointly within the same group, which gives rise to a way of parsimoniously modeling multivariate functional data. In this paper, a novel grouped multivariate functional regression model with a new regularization approach termed "coefficient shape alignment" is developed to tackle the potential homogeneity of different functional covariates. The modeling procedure includes two main steps: first detect the unknown grouping structure with the new regularization approach to aggregate covariates into disjoint groups; and then the grouped multivariate functional regression model is established based on the detected grouping structure. In this new grouped model, the coefficient functions of covariates in the same homogeneous group sh
    
[^2]: 交通图未调整的 Langevin 算法：学习和离散化扰动采样器

    Transport map unadjusted Langevin algorithms: learning and discretizing perturbed samplers. (arXiv:2302.07227v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.07227](http://arxiv.org/abs/2302.07227)

    本研究提出了交通图未调整的 Langevin 算法 (ULA) 和 Riemann 流形 Langevin 动力学 (RMLD)，通过应用交通图可以加速 Langevin 动力学的收敛，并提供了学习度量和扰动的新思路。

    

    Langevin 动力学被广泛用于抽样高维、非高斯分布，其密度已知但常数未知。特别感兴趣的是未修正的 Langevin 算法 (ULA)，它直接离散化 Langevin 动力学以估计目标分布上的期望。我们研究了使用交通图来近似标准化目标分布的方法，以预处理和加速 Langevin 动力学的收敛。我们展示了在连续时间下，当将交通图应用于 Langevin 动力学时，结果是具有由交通图定义的度量的 Riemann 流形 Langevin 动力学（RMLD）。我们还展示了将交通图应用于不可逆扰动的 ULA 会产生原动力学的几何信息不可逆扰动 （GiIrr）。这些联系表明了学习度量和扰动的更系统的方法，并提供了描述 RMLD 的替代离散化方法。

    Langevin dynamics are widely used in sampling high-dimensional, non-Gaussian distributions whose densities are known up to a normalizing constant. In particular, there is strong interest in unadjusted Langevin algorithms (ULA), which directly discretize Langevin dynamics to estimate expectations over the target distribution. We study the use of transport maps that approximately normalize a target distribution as a way to precondition and accelerate the convergence of Langevin dynamics. We show that in continuous time, when a transport map is applied to Langevin dynamics, the result is a Riemannian manifold Langevin dynamics (RMLD) with metric defined by the transport map. We also show that applying a transport map to an irreversibly-perturbed ULA results in a geometry-informed irreversible perturbation (GiIrr) of the original dynamics. These connections suggest more systematic ways of learning metrics and perturbations, and also yield alternative discretizations of the RMLD described b
    

