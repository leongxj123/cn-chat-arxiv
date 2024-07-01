# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Rate of Kernel Regression in Large Dimensions.](http://arxiv.org/abs/2309.04268) | 该论文提出了一种针对大维度数据的核回归的最优比率，通过使用Mendelson复杂性和度量熵来刻画其上界和最小化下界。此外，研究还发现最优比率随着维度与样本大小关系的变化呈现出多次下降的行为。 |

# 详细

[^1]: 大维度情况下核回归的最优比率

    Optimal Rate of Kernel Regression in Large Dimensions. (arXiv:2309.04268v1 [stat.ML])

    [http://arxiv.org/abs/2309.04268](http://arxiv.org/abs/2309.04268)

    该论文提出了一种针对大维度数据的核回归的最优比率，通过使用Mendelson复杂性和度量熵来刻画其上界和最小化下界。此外，研究还发现最优比率随着维度与样本大小关系的变化呈现出多次下降的行为。

    

    我们对大维度数据（样本大小$n$与样本维度$d$的关系为多项式，即$n\asymp d^{\gamma}$，其中$\gamma>0$）的核回归进行了研究。我们首先通过Mendelson复杂性$\varepsilon_{n}^{2}$和度量熵$\bar{\varepsilon}_{n}^{2}$来建立一个通用工具，用于刻画大维度数据的核回归的上界和最小化下界。当目标函数属于与$\mathbb{S}^{d}$上定义的（一般）内积模型相关联的RKHS时，我们利用这个新工具来展示核回归的过量风险的最小化率是$n^{-1/2}$，当$n\asymp d^{\gamma}$，其中$\gamma=2, 4, 6, 8, \cdots$。然后我们进一步确定了对于所有$\gamma>0$，核回归过量风险的最优比率，并发现随着$\gamma$的变化，最优比率的曲线展现出几个新现象，包括多次下降行为。

    We perform a study on kernel regression for large-dimensional data (where the sample size $n$ is polynomially depending on the dimension $d$ of the samples, i.e., $n\asymp d^{\gamma}$ for some $\gamma >0$ ). We first build a general tool to characterize the upper bound and the minimax lower bound of kernel regression for large dimensional data through the Mendelson complexity $\varepsilon_{n}^{2}$ and the metric entropy $\bar{\varepsilon}_{n}^{2}$ respectively. When the target function falls into the RKHS associated with a (general) inner product model defined on $\mathbb{S}^{d}$, we utilize the new tool to show that the minimax rate of the excess risk of kernel regression is $n^{-1/2}$ when $n\asymp d^{\gamma}$ for $\gamma =2, 4, 6, 8, \cdots$. We then further determine the optimal rate of the excess risk of kernel regression for all the $\gamma>0$ and find that the curve of optimal rate varying along $\gamma$ exhibits several new phenomena including the {\it multiple descent behavior
    

