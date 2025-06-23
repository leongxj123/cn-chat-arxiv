# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dimension free ridge regression](https://arxiv.org/abs/2210.08571) | 该论文旨在超越比例渐近情况，重新审视对高维数据进行岭回归，允许特征向量是高维甚至无限维的，为统计学中的自然设置提供了新的研究方向。 |

# 详细

[^1]: 无维度岭回归

    Dimension free ridge regression

    [https://arxiv.org/abs/2210.08571](https://arxiv.org/abs/2210.08571)

    该论文旨在超越比例渐近情况，重新审视对高维数据进行岭回归，允许特征向量是高维甚至无限维的，为统计学中的自然设置提供了新的研究方向。

    

    随机矩阵理论已成为高维统计学和理论机器学习中非常有用的工具。然而，随机矩阵理论主要集中在比例渐近情况下，其中列数与数据矩阵的行数成比例增长。这在统计学中并不总是最自然的设置，其中列对应协变量，行对应样本。为了超越比例渐近，我们重新审视了对独立同分布数据$(x_i, y_i)$，$i\le n$进行岭回归（$\ell_2$-惩罚最小二乘），其中$x_i$为特征向量，$y_i = \beta^\top x_i +\epsilon_i \in\mathbb{R}$为响应。我们允许特征向量是高维的，甚至是无限维的，此时它属于可分Hilbert空间，并且假设$z_i := \Sigma^{-1/2}x_i$具有独立同分布的条目，或者满足某种凸集中性质。

    arXiv:2210.08571v2 Announce Type: replace-cross  Abstract: Random matrix theory has become a widely useful tool in high-dimensional statistics and theoretical machine learning. However, random matrix theory is largely focused on the proportional asymptotics in which the number of columns grows proportionally to the number of rows of the data matrix. This is not always the most natural setting in statistics where columns correspond to covariates and rows to samples. With the objective to move beyond the proportional asymptotics, we revisit ridge regression ($\ell_2$-penalized least squares) on i.i.d. data $(x_i, y_i)$, $i\le n$, where $x_i$ is a feature vector and $y_i = \beta^\top x_i +\epsilon_i \in\mathbb{R}$ is a response. We allow the feature vector to be high-dimensional, or even infinite-dimensional, in which case it belongs to a separable Hilbert space, and assume either $z_i := \Sigma^{-1/2}x_i$ to have i.i.d. entries, or to satisfy a certain convex concentration property. With
    

