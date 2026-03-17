# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-dependent PDE solver based on signature kernels](https://arxiv.org/abs/2403.11738) | 该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。 |
| [^2] | [Measure transfer via stochastic slicing and matching.](http://arxiv.org/abs/2307.05705) | 本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。 |

# 详细

[^1]: 基于特征核的路径依赖PDE求解器

    A path-dependent PDE solver based on signature kernels

    [https://arxiv.org/abs/2403.11738](https://arxiv.org/abs/2403.11738)

    该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。

    

    我们开发了一种基于特征核的路径依赖PDE（PPDE）的收敛证明求解器。我们的数值方案利用了特征核，这是最近在路径空间上引入的一类核。具体来说，我们通过在符号再生核希尔伯特空间（RKHS）中近似PPDE的解来解决一个最优恢复问题，该空间受到在有限集合的远程路径上满足PPDE约束的元素的约束。在线性情况下，我们证明了优化具有唯一的闭式解，其以远程路径处的特征核评估的形式表示。我们证明了所提出方案的一致性，保证在远程点数增加时收敛到PPDE解。最后，我们提供了几个数值例子，尤其是在粗糙波动率下的期权定价背景下。我们的数值方案构成了一种替代性的蒙特卡洛方法的有效替代方案。

    arXiv:2403.11738v1 Announce Type: cross  Abstract: We develop a provably convergent kernel-based solver for path-dependent PDEs (PPDEs). Our numerical scheme leverages signature kernels, a recently introduced class of kernels on path-space. Specifically, we solve an optimal recovery problem by approximating the solution of a PPDE with an element of minimal norm in the signature reproducing kernel Hilbert space (RKHS) constrained to satisfy the PPDE at a finite collection of collocation paths. In the linear case, we show that the optimisation has a unique closed-form solution expressed in terms of signature kernel evaluations at the collocation paths. We prove consistency of the proposed scheme, guaranteeing convergence to the PPDE solution as the number of collocation points increases. Finally, several numerical examples are presented, in particular in the context of option pricing under rough volatility. Our numerical scheme constitutes a valid alternative to the ubiquitous Monte Carl
    
[^2]: 通过随机切片和配准进行测量转移

    Measure transfer via stochastic slicing and matching. (arXiv:2307.05705v1 [math.NA])

    [http://arxiv.org/abs/2307.05705](http://arxiv.org/abs/2307.05705)

    本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。

    

    本论文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案。类似于切片Wasserstein距离，这些方案受益于一维最优输运问题的闭式解的可用性和相关计算优势。尽管这些方案已经在数据科学应用中取得了成功，但关于它们的收敛性的结果不太多。本文的主要贡献是对随机切片和配准方案提供了几乎必然收敛的证明。该证明建立在将其解释为Wasserstein空间上的随机梯度下降方案的基础之上。同时还展示了关于逐步图像变形的数值示例。

    This paper studies iterative schemes for measure transfer and approximation problems, which are defined through a slicing-and-matching procedure. Similar to the sliced Wasserstein distance, these schemes benefit from the availability of closed-form solutions for the one-dimensional optimal transport problem and the associated computational advantages. While such schemes have already been successfully utilized in data science applications, not too many results on their convergence are available. The main contribution of this paper is an almost sure convergence proof for stochastic slicing-and-matching schemes. The proof builds on an interpretation as a stochastic gradient descent scheme on the Wasserstein space. Numerical examples on step-wise image morphing are demonstrated as well.
    

