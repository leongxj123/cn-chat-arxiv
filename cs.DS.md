# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Decision Trees for Separable Objectives: Pushing the Limits of Dynamic Programming.](http://arxiv.org/abs/2305.19706) | 本研究提出了一种通用的动态规划方法来优化任何组合的可分离目标和约束条件，这种方法在可扩展性方面比通用求解器表现得更好。 |

# 详细

[^1]: 可分目标的最优决策树：推动动态规划的极限

    Optimal Decision Trees for Separable Objectives: Pushing the Limits of Dynamic Programming. (arXiv:2305.19706v1 [cs.LG])

    [http://arxiv.org/abs/2305.19706](http://arxiv.org/abs/2305.19706)

    本研究提出了一种通用的动态规划方法来优化任何组合的可分离目标和约束条件，这种方法在可扩展性方面比通用求解器表现得更好。

    

    决策树的全局优化在准确性，大小和人类可理解性方面表现出良好的前景。然而，许多方法仍然依赖于通用求解器，可扩展性仍然是一个问题。动态规划方法已被证明具有更好的可扩展性，因为它们通过将子树作为独立的子问题解决来利用树结构。然而，这仅适用于可以分别优化子树的任务。我们详细研究了这种关系，并展示了实现这种可分离约束和目标任意组合的动态规划方法。在四个应用领域的实验表明了这种方法的普适性，同时也比通用求解器具有更好的可扩展性。

    Global optimization of decision trees has shown to be promising in terms of accuracy, size, and consequently human comprehensibility. However, many of the methods used rely on general-purpose solvers for which scalability remains an issue. Dynamic programming methods have been shown to scale much better because they exploit the tree structure by solving subtrees as independent subproblems. However, this only works when an objective can be optimized separately for subtrees. We explore this relationship in detail and show necessary and sufficient conditions for such separability and generalize previous dynamic programming approaches into a framework that can optimize any combination of separable objectives and constraints. Experiments on four application domains show the general applicability of this framework, while outperforming the scalability of general-purpose solvers by a large margin.
    

