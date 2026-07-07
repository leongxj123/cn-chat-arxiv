# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Accelerated Stochastic Algorithm for Solving the Optimal Transport Problem.](http://arxiv.org/abs/2203.00813) | 本文提出了一种能够用较低的计算复杂度解决最优输运问题的加速随机算法。 |

# 详细

[^1]: 一种用于解决最优输运问题的加速随机算法

    An Accelerated Stochastic Algorithm for Solving the Optimal Transport Problem. (arXiv:2203.00813v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.00813](http://arxiv.org/abs/2203.00813)

    本文提出了一种能够用较低的计算复杂度解决最优输运问题的加速随机算法。

    

    本文提出了一种原始-对偶加速随机梯度下降与方差约减算法(PDASGD)，用于解决线性约束优化问题。 PDASGD可应用于解决离散最优输运（OT）问题，并具有已知的最佳计算复杂度-$\widetilde{\mathcal{O}}(n^2/\epsilon)$，其中$n$是原子数，$\epsilon> 0$是精度。 本文还讨论了使得PDASGD具有较低的计算复杂度的条件，来解决线性约束优化问题。 数值实验表明，该算法在解决OT问题时可以将复杂度的速率提高了$\widetilde{\mathcal{O}}(\sqrt{n})$。

    A primal-dual accelerated stochastic gradient descent with variance reduction algorithm (PDASGD) is proposed to solve linear-constrained optimization problems. PDASGD could be applied to solve the discrete optimal transport (OT) problem and enjoys the best-known computational complexity -$\widetilde{\mathcal{O}}(n^2/\epsilon)$, where $n$ is the number of atoms, and $\epsilon>0$ is the accuracy. In the literature, some primal-dual accelerated first-order algorithms, e.g., APDAGD, have been proposed and have the order of $\widetilde{\mathcal{O}}(n^{2.5}/\epsilon)$ for solving the OT problem. To understand why our proposed algorithm could improve the rate by a factor of $\widetilde{\mathcal{O}}(\sqrt{n})$, the conditions under which our stochastic algorithm has a lower order of computational complexity for solving linear-constrained optimization problems are discussed. It is demonstrated that the OT problem could satisfy the aforementioned conditions. Numerical experiments demonstrate s
    

