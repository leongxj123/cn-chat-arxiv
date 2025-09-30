# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Explicit Second-Order Min-Max Optimization Methods with Optimal Convergence Guarantee.](http://arxiv.org/abs/2210.12860) | 本文提出了一种具有最优收敛保证的显式二阶最小最大优化方法，用于解决凸凹无约束最小最大优化问题。该方法利用二阶信息加速额外梯度方法，并且在迭代过程中保持在有界集内，达到了与理论下界相匹配的收敛速度。 |

# 详细

[^1]: 具有最优收敛保证的显式二阶最小最大优化方法

    Explicit Second-Order Min-Max Optimization Methods with Optimal Convergence Guarantee. (arXiv:2210.12860v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2210.12860](http://arxiv.org/abs/2210.12860)

    本文提出了一种具有最优收敛保证的显式二阶最小最大优化方法，用于解决凸凹无约束最小最大优化问题。该方法利用二阶信息加速额外梯度方法，并且在迭代过程中保持在有界集内，达到了与理论下界相匹配的收敛速度。

    

    本文提出并分析了一种精确和不精确正则化牛顿型方法，用于求解凸凹无约束最小最大优化问题的全局鞍点。与一阶方法相比，我们对于二阶最小最大优化方法的理解相对较少，因为利用二阶信息获得全局收敛速度更加复杂。在本文中，我们研究了如何利用二阶信息加速额外梯度方法，即使在不精确的情况下也能实现。具体而言，我们证明了所提出的算法生成的迭代保持在有界集内，并且平均迭代收敛到一个 $\epsilon$-鞍点，所需迭代次数为 $O(\epsilon^{-2/3})$，其中使用了受限间隙函数。我们的算法与该领域已经建立的理论下界相匹配，而且我们的分析提供了一种简单直观的二阶方法收敛分析，不需要任何有界性要求。最后，我们提出了一个

    We propose and analyze exact and inexact regularized Newton-type methods for finding a global saddle point of \emph{convex-concave} unconstrained min-max optimization problems. Compared to first-order methods, our understanding of second-order methods for min-max optimization is relatively limited, as obtaining global rates of convergence with second-order information is much more involved. In this paper, we examine how second-order information can be used to speed up extra-gradient methods, even under inexactness. Specifically, we show that the proposed algorithms generate iterates that remain within a bounded set and the averaged iterates converge to an $\epsilon$-saddle point within $O(\epsilon^{-2/3})$ iterations in terms of a restricted gap function. Our algorithms match the theoretically established lower bound in this context and our analysis provides a simple and intuitive convergence analysis for second-order methods without any boundedness requirements. Finally, we present a 
    

