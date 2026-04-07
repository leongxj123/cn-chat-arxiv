# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^2] | [Sparse Gaussian Graphical Models with Discrete Optimization: Computational and Statistical Perspectives.](http://arxiv.org/abs/2307.09366) | 本文提出了基于离散优化的稀疏高斯图模型学习问题的新方法，并提供了大规模求解器来获取良好的原始解。 |

# 详细

[^1]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^2]: 稀疏高斯图模型的离散优化：计算和统计角度

    Sparse Gaussian Graphical Models with Discrete Optimization: Computational and Statistical Perspectives. (arXiv:2307.09366v1 [cs.LG])

    [http://arxiv.org/abs/2307.09366](http://arxiv.org/abs/2307.09366)

    本文提出了基于离散优化的稀疏高斯图模型学习问题的新方法，并提供了大规模求解器来获取良好的原始解。

    

    我们考虑了学习基于无向高斯图模型的稀疏图的问题，这是统计机器学习中的一个关键问题。给定来自具有p个变量的多元高斯分布的n个样本，目标是估计p×p的逆协方差矩阵（也称为精度矩阵），假设它是稀疏的（即具有少数非零条目）。我们提出了GraphL0BnB这一新的估计方法，它基于伪似然函数的l0惩罚版本，而大多数早期方法都是基于l1松弛。我们的估计方法可以被形式化为一个凸混合整数规划（MIP），使用现成的商用求解器在大规模计算时可能很难计算。为了解决MIP问题，我们提出了一个定制的非线性分支定界（BnB）框架，用于使用定制的一阶方法来解决节点放松问题。作为我们BnB框架的副产品，我们提出了用于获得独立兴趣的良好原始解的大规模求解器。

    We consider the problem of learning a sparse graph underlying an undirected Gaussian graphical model, a key problem in statistical machine learning. Given $n$ samples from a multivariate Gaussian distribution with $p$ variables, the goal is to estimate the $p \times p$ inverse covariance matrix (aka precision matrix), assuming it is sparse (i.e., has a few nonzero entries). We propose GraphL0BnB, a new estimator based on an $\ell_0$-penalized version of the pseudolikelihood function, while most earlier approaches are based on the $\ell_1$-relaxation. Our estimator can be formulated as a convex mixed integer program (MIP) which can be difficult to compute at scale using off-the-shelf commercial solvers. To solve the MIP, we propose a custom nonlinear branch-and-bound (BnB) framework that solves node relaxations with tailored first-order methods. As a by-product of our BnB framework, we propose large-scale solvers for obtaining good primal solutions that are of independent interest. We d
    

