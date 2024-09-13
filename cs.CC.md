# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Computational Complexity of Finding Stationary Points in Non-Convex Optimization.](http://arxiv.org/abs/2310.09157) | 本文研究了在非凸优化中找到光滑目标函数的近似稳定点的计算复杂性和查询复杂性，并给出了相应的结果。对于$d=2$的情况，提供了一种零阶算法，只需要少量的函数值查询即可找到$\varepsilon$-近似稳定点。 |

# 详细

[^1]: 寻找非凸优化中的稳定点的计算复杂性

    The Computational Complexity of Finding Stationary Points in Non-Convex Optimization. (arXiv:2310.09157v1 [math.OC])

    [http://arxiv.org/abs/2310.09157](http://arxiv.org/abs/2310.09157)

    本文研究了在非凸优化中找到光滑目标函数的近似稳定点的计算复杂性和查询复杂性，并给出了相应的结果。对于$d=2$的情况，提供了一种零阶算法，只需要少量的函数值查询即可找到$\varepsilon$-近似稳定点。

    

    寻找非凸但光滑目标函数$f$在无限制的$d$维域上的近似稳定点，即梯度近似为零的点，是经典非凸优化中最基本的问题之一。然而，当问题的维度$d$与近似误差独立时，这个问题的计算复杂性和查询复杂性仍不十分清楚。在本文中，我们展示了以下计算复杂性和查询复杂性结果：1.在无限制的域中寻找近似稳定点的问题是PLS完全问题。2.对于$d=2$，我们提供了一种零阶算法，用于寻找$\varepsilon$-近似稳定点，只需要对目标函数进行最多$O(1/\varepsilon)$次函数值查询。3.我们证明当$d=2$时，任何算法至少需要$\Omega(1/\varepsilon)$次对目标函数和/或梯度的查询来找到$\varepsilon$-近似稳定点。

    Finding approximate stationary points, i.e., points where the gradient is approximately zero, of non-convex but smooth objective functions $f$ over unrestricted $d$-dimensional domains is one of the most fundamental problems in classical non-convex optimization. Nevertheless, the computational and query complexity of this problem are still not well understood when the dimension $d$ of the problem is independent of the approximation error. In this paper, we show the following computational and query complexity results:  1. The problem of finding approximate stationary points over unrestricted domains is PLS-complete.  2. For $d = 2$, we provide a zero-order algorithm for finding $\varepsilon$-approximate stationary points that requires at most $O(1/\varepsilon)$ value queries to the objective function.  3. We show that any algorithm needs at least $\Omega(1/\varepsilon)$ queries to the objective function and/or its gradient to find $\varepsilon$-approximate stationary points when $d=2$.
    

