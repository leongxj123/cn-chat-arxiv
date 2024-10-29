# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-dependent PDE solver based on signature kernels](https://arxiv.org/abs/2403.11738) | 该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。 |
| [^2] | [Multiscale Hodge Scattering Networks for Data Analysis](https://arxiv.org/abs/2311.10270) | 提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。 |
| [^3] | [A first-order augmented Lagrangian method for constrained minimax optimization.](http://arxiv.org/abs/2301.02060) | 本文提出了一种一阶增广拉格朗日方法来解决约束极小极大问题，其操作复杂度为 ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$。 |

# 详细

[^1]: 基于特征核的路径依赖PDE求解器

    A path-dependent PDE solver based on signature kernels

    [https://arxiv.org/abs/2403.11738](https://arxiv.org/abs/2403.11738)

    该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。

    

    我们开发了一种基于特征核的路径依赖PDE（PPDE）的收敛证明求解器。我们的数值方案利用了特征核，这是最近在路径空间上引入的一类核。具体来说，我们通过在符号再生核希尔伯特空间（RKHS）中近似PPDE的解来解决一个最优恢复问题，该空间受到在有限集合的远程路径上满足PPDE约束的元素的约束。在线性情况下，我们证明了优化具有唯一的闭式解，其以远程路径处的特征核评估的形式表示。我们证明了所提出方案的一致性，保证在远程点数增加时收敛到PPDE解。最后，我们提供了几个数值例子，尤其是在粗糙波动率下的期权定价背景下。我们的数值方案构成了一种替代性的蒙特卡洛方法的有效替代方案。

    arXiv:2403.11738v1 Announce Type: cross  Abstract: We develop a provably convergent kernel-based solver for path-dependent PDEs (PPDEs). Our numerical scheme leverages signature kernels, a recently introduced class of kernels on path-space. Specifically, we solve an optimal recovery problem by approximating the solution of a PPDE with an element of minimal norm in the signature reproducing kernel Hilbert space (RKHS) constrained to satisfy the PPDE at a finite collection of collocation paths. In the linear case, we show that the optimisation has a unique closed-form solution expressed in terms of signature kernel evaluations at the collocation paths. We prove consistency of the proposed scheme, guaranteeing convergence to the PPDE solution as the number of collocation points increases. Finally, several numerical examples are presented, in particular in the context of option pricing under rough volatility. Our numerical scheme constitutes a valid alternative to the ubiquitous Monte Carl
    
[^2]: 用于数据分析的多尺度霍奇散射网络

    Multiscale Hodge Scattering Networks for Data Analysis

    [https://arxiv.org/abs/2311.10270](https://arxiv.org/abs/2311.10270)

    提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。

    

    我们提出了一种新的散射网络，用于在单纯复合仿射上测量的信号，称为\emph{多尺度霍奇散射网络}（MHSNs）。我们的构造基于单纯复合仿射上的多尺度基础词典，即$\kappa$-GHWT和$\kappa$-HGLET，我们最近为给定单纯复合仿射中的维度$\kappa \in \mathbb{N}$推广了基于节点的广义哈-沃什变换（GHWT）和分层图拉普拉斯特征变换（HGLET）。$\kappa$-GHWT和$\kappa$-HGLET都形成冗余集合（即词典）的多尺度基础向量和给定信号的相应扩展系数。我们的MHSNs使用类似于卷积神经网络（CNN）的分层结构来级联词典系数模的矩。所得特征对单纯复合仿射的重新排序不变（即节点排列的置换

    arXiv:2311.10270v2 Announce Type: replace  Abstract: We propose new scattering networks for signals measured on simplicial complexes, which we call \emph{Multiscale Hodge Scattering Networks} (MHSNs). Our construction is based on multiscale basis dictionaries on simplicial complexes, i.e., the $\kappa$-GHWT and $\kappa$-HGLET, which we recently developed for simplices of dimension $\kappa \in \mathbb{N}$ in a given simplicial complex by generalizing the node-based Generalized Haar-Walsh Transform (GHWT) and Hierarchical Graph Laplacian Eigen Transform (HGLET). The $\kappa$-GHWT and the $\kappa$-HGLET both form redundant sets (i.e., dictionaries) of multiscale basis vectors and the corresponding expansion coefficients of a given signal. Our MHSNs use a layered structure analogous to a convolutional neural network (CNN) to cascade the moments of the modulus of the dictionary coefficients. The resulting features are invariant to reordering of the simplices (i.e., node permutation of the u
    
[^3]: 一种用于约束极小极大优化问题的一阶增广拉格朗日方法

    A first-order augmented Lagrangian method for constrained minimax optimization. (arXiv:2301.02060v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2301.02060](http://arxiv.org/abs/2301.02060)

    本文提出了一种一阶增广拉格朗日方法来解决约束极小极大问题，其操作复杂度为 ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$。

    

    本文研究了一类约束极小极大问题。特别地，我们提出了一种一阶增广拉格朗日方法来解决这些问题，其子问题被发现是一个更简单的结构化极小极大问题，并且可以通过作者在 [26] 中最近开发的一阶方法来适当地解决。在一些适当的假设下，为了找到约束极小极大问题的一个 $\varepsilon$-KKT 解，该方法的操作复杂度为 ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$，该复杂度是由基本操作测量得到的。

    In this paper we study a class of constrained minimax problems. In particular, we propose a first-order augmented Lagrangian method for solving them, whose subproblems turn out to be a much simpler structured minimax problem and are suitably solved by a first-order method recently developed in [26] by the authors. Under some suitable assumptions, an \emph{operation complexity} of ${\cal O}(\varepsilon^{-4}\log\varepsilon^{-1})$, measured by its fundamental operations, is established for the first-order augmented Lagrangian method for finding an $\varepsilon$-KKT solution of the constrained minimax problems.
    

