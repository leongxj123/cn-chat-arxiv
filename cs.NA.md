# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite Expression Method for Solving High-Dimensional Partial Differential Equations.](http://arxiv.org/abs/2206.10121) | 本文介绍了一种名为有限表达方法（FEX）的新方法，用于在具有有限个解析表达式的函数空间中寻找高维偏微分方程的近似解。通过使用深度强化学习方法将FEX应用于各种高维偏微分方程，可以实现高度准确的求解，并且避免了维度灾难。这种有限解析表达式的近似解还可以提供对真实偏微分方程解的可解释洞察。 |

# 详细

[^1]: 用于求解高维偏微分方程的有限表达方法

    Finite Expression Method for Solving High-Dimensional Partial Differential Equations. (arXiv:2206.10121v3 [math.NA] UPDATED)

    [http://arxiv.org/abs/2206.10121](http://arxiv.org/abs/2206.10121)

    本文介绍了一种名为有限表达方法（FEX）的新方法，用于在具有有限个解析表达式的函数空间中寻找高维偏微分方程的近似解。通过使用深度强化学习方法将FEX应用于各种高维偏微分方程，可以实现高度准确的求解，并且避免了维度灾难。这种有限解析表达式的近似解还可以提供对真实偏微分方程解的可解释洞察。

    

    设计高效准确的高维偏微分方程数值求解器仍然是计算科学和工程中一个具有挑战性和重要性的课题，主要是由于在设计能够按维数进行扩展的数值方案中存在“维度灾难”。本文介绍了一种新的方法，该方法在具有有限个解析表达式的函数空间中寻找近似的偏微分方程解，因此将该方法命名为有限表达方法（FEX）。在近似理论中证明了FEX可以避免维度灾难。作为概念证明，本文提出了一种深度强化学习方法，用于在不同维度上实现FEX求解各种高维偏微分方程，实现了高甚至机器精度，并具有多项式维度的记忆复杂度和可操作的时间复杂度。具有有限解析表达式的近似解还为地面真实偏微分方程解提供了可解释的洞察。

    Designing efficient and accurate numerical solvers for high-dimensional partial differential equations (PDEs) remains a challenging and important topic in computational science and engineering, mainly due to the "curse of dimensionality" in designing numerical schemes that scale in dimension. This paper introduces a new methodology that seeks an approximate PDE solution in the space of functions with finitely many analytic expressions and, hence, this methodology is named the finite expression method (FEX). It is proved in approximation theory that FEX can avoid the curse of dimensionality. As a proof of concept, a deep reinforcement learning method is proposed to implement FEX for various high-dimensional PDEs in different dimensions, achieving high and even machine accuracy with a memory complexity polynomial in dimension and an amenable time complexity. An approximate solution with finite analytic expressions also provides interpretable insights into the ground truth PDE solution, w
    

