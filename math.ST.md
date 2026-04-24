# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence Rates for Non-Log-Concave Sampling and Log-Partition Estimation.](http://arxiv.org/abs/2303.03237) | 非对数凹势V的高维采样速率可以在一些条件下实现与凸函数相同的收敛速率。 |

# 详细

[^1]: 非对数凹采样和对数分区估计的收敛速率

    Convergence Rates for Non-Log-Concave Sampling and Log-Partition Estimation. (arXiv:2303.03237v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.03237](http://arxiv.org/abs/2303.03237)

    非对数凹势V的高维采样速率可以在一些条件下实现与凸函数相同的收敛速率。

    

    从吉布斯分布$p(x)\propto\exp(-V(x)/\epsilon)$中采样并计算其对数分区函数是统计学、机器学习和统计物理中的基本任务。然而，虽然有效的算法已知于凸势函数$V$，但非凸情况下的情况要困难得多，算法必然在最坏情况下受到维度灾难的困扰。最近，已经证明在适当的条件下，高维采样非对数凹势V的速率也可以达到同样快的速度。本文对这些结果进行了回顾，并强调了领域中的一些开放问题。

    Sampling from Gibbs distributions $p(x) \propto \exp(-V(x)/\varepsilon)$ and computing their log-partition function are fundamental tasks in statistics, machine learning, and statistical physics. However, while efficient algorithms are known for convex potentials $V$, the situation is much more difficult in the non-convex case, where algorithms necessarily suffer from the curse of dimensionality in the worst case. For optimization, which can be seen as a low-temperature limit of sampling, it is known that smooth functions $V$ allow faster convergence rates. Specifically, for $m$-times differentiable functions in $d$ dimensions, the optimal rate for algorithms with $n$ function evaluations is known to be $O(n^{-m/d})$, where the constant can potentially depend on $m, d$ and the function to be optimized. Hence, the curse of dimensionality can be alleviated for smooth functions at least in terms of the convergence rate. Recently, it has been shown that similarly fast rates can also be ach
    

