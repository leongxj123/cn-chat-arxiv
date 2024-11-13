# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Signal reconstruction using determinantal sampling.](http://arxiv.org/abs/2310.09437) | 本研究提出了使用行列式抽样进行信号重建的方法，在有限数量的随机节点评估中近似表示方可积函数，实现了快速收敛和更高的适应性正则性。 |
| [^2] | [A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints.](http://arxiv.org/abs/2304.03641) | 本文提出了一种新的块坐标下降方法OBCD，用于解决具有正交约束的一般非光滑组合问题。 OBCD是一种可行的方法，具有低的计算复杂性，并且获得严格的收敛保证。 |

# 详细

[^1]: 使用行列式抽样进行信号重建

    Signal reconstruction using determinantal sampling. (arXiv:2310.09437v1 [stat.ML])

    [http://arxiv.org/abs/2310.09437](http://arxiv.org/abs/2310.09437)

    本研究提出了使用行列式抽样进行信号重建的方法，在有限数量的随机节点评估中近似表示方可积函数，实现了快速收敛和更高的适应性正则性。

    

    我们研究了从随机节点的有限数量评估中近似表示一个方可积函数的问题，其中随机节点的选择依据是一个精心选择的分布。当函数被假设属于再生核希尔伯特空间（RKHS）时，这尤为相关。本研究提出了将基于两种可能的节点概率分布的几个自然有限维逼近方法相结合。这些概率分布与行列式点过程相关，并利用RKHS的核函数来优化在随机设计中的RKHS适应性正则性。虽然先前的行列式抽样工作依赖于RKHS范数，我们证明了在$L^2$范数下的均方保证。我们表明，行列式点过程及其混合体可以产生快速收敛速度。我们的结果还揭示了当假设更多的平滑性时收敛速度如何变化，这种现象被称为超收敛。此外，行列式抽样推广了从Christoffel函数进行i.i.d.抽样的方法。

    We study the approximation of a square-integrable function from a finite number of evaluations on a random set of nodes according to a well-chosen distribution. This is particularly relevant when the function is assumed to belong to a reproducing kernel Hilbert space (RKHS). This work proposes to combine several natural finite-dimensional approximations based two possible probability distributions of nodes. These distributions are related to determinantal point processes, and use the kernel of the RKHS to favor RKHS-adapted regularity in the random design. While previous work on determinantal sampling relied on the RKHS norm, we prove mean-square guarantees in $L^2$ norm. We show that determinantal point processes and mixtures thereof can yield fast convergence rates. Our results also shed light on how the rate changes as more smoothness is assumed, a phenomenon known as superconvergence. Besides, determinantal sampling generalizes i.i.d. sampling from the Christoffel function which is
    
[^2]: 一种用于正交约束下的非光滑组合优化的块坐标下降方法

    A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints. (arXiv:2304.03641v1 [math.OC])

    [http://arxiv.org/abs/2304.03641](http://arxiv.org/abs/2304.03641)

    本文提出了一种新的块坐标下降方法OBCD，用于解决具有正交约束的一般非光滑组合问题。 OBCD是一种可行的方法，具有低的计算复杂性，并且获得严格的收敛保证。

    

    具有正交约束的非光滑组合优化在统计学习和数据科学中有广泛的应用。由于其非凸性和非光滑性质，该问题通常很难求解。现有的解决方案受到以下一个或多个限制的限制：（i）它们是需要每次迭代高计算成本的全梯度方法；（ii）它们无法解决一般的非光滑组合问题；（iii）它们是不可行方法，并且只能在极限点处实现解的可行性；（iv）它们缺乏严格的收敛保证；（v）它们只能获得关键点的弱最优性。在本文中，我们提出了一种新的块坐标下降方法OBCD，用于解决正交约束下的一般非光滑组合问题。OBCD是一种可行的方法，具有低的计算复杂性。在每次迭代中，我们的算法会更新...

    Nonsmooth composite optimization with orthogonality constraints has a broad spectrum of applications in statistical learning and data science. However, this problem is generally challenging to solve due to its non-convex and non-smooth nature. Existing solutions are limited by one or more of the following restrictions: (i) they are full gradient methods that require high computational costs in each iteration; (ii) they are not capable of solving general nonsmooth composite problems; (iii) they are infeasible methods and can only achieve the feasibility of the solution at the limit point; (iv) they lack rigorous convergence guarantees; (v) they only obtain weak optimality of critical points. In this paper, we propose \textit{\textbf{OBCD}}, a new Block Coordinate Descent method for solving general nonsmooth composite problems under Orthogonality constraints. \textit{\textbf{OBCD}} is a feasible method with low computation complexity footprints. In each iteration, our algorithm updates $
    

