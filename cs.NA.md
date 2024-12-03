# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Correction to "Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations"](https://arxiv.org/abs/2402.08711) | 修正了《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的错误局部误差估计，提出了一种方法来分析数值离散遍历SDE的Wasserstein-2距离的非渐近保证，并解决了实践中维度依赖性的问题。 |
| [^2] | [A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints.](http://arxiv.org/abs/2304.03641) | 本文提出了一种新的块坐标下降方法OBCD，用于解决具有正交约束的一般非光滑组合问题。 OBCD是一种可行的方法，具有低的计算复杂性，并且获得严格的收敛保证。 |

# 详细

[^1]: 《对于数值逼近遍历SDE的分布的Wasserstein距离估计》修正

    Correction to "Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations"

    [https://arxiv.org/abs/2402.08711](https://arxiv.org/abs/2402.08711)

    修正了《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的错误局部误差估计，提出了一种方法来分析数值离散遍历SDE的Wasserstein-2距离的非渐近保证，并解决了实践中维度依赖性的问题。

    

    本文对San-Serna和Zygalakis的《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的非渐近保证数值离散分析方法进行了修正。他们分析了UBU积分器，该积分器是二阶强型的，并且每个步骤只需要一次梯度评估，从而得到了理想的非渐近保证，特别是在Wasserstein-2距离中到达离目标分布 $\epsilon > 0$ 的距离仅需 $\mathcal{O}(d^{1/4}\epsilon^{-1/2})$ 步。然而，Sanz-Serna和Zygalakis (2021)中的局部误差估计存在错误，在实践中需要更强的假设才能实现这些复杂度估计。本文解决了理论与实践中观察到的许多应用场景中的维度依赖性。

    arXiv:2402.08711v1 Announce Type: cross Abstract: A method for analyzing non-asymptotic guarantees of numerical discretizations of ergodic SDEs in Wasserstein-2 distance is presented by Sanz-Serna and Zygalakis in ``Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations". They analyze the UBU integrator which is strong order two and only requires one gradient evaluation per step, resulting in desirable non-asymptotic guarantees, in particular $\mathcal{O}(d^{1/4}\epsilon^{-1/2})$ steps to reach a distance of $\epsilon > 0$ in Wasserstein-2 distance away from the target distribution. However, there is a mistake in the local error estimates in Sanz-Serna and Zygalakis (2021), in particular, a stronger assumption is needed to achieve these complexity estimates. This note reconciles the theory with the dimension dependence observed in practice in many applications of interest.
    
[^2]: 一种用于正交约束下的非光滑组合优化的块坐标下降方法

    A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints. (arXiv:2304.03641v1 [math.OC])

    [http://arxiv.org/abs/2304.03641](http://arxiv.org/abs/2304.03641)

    本文提出了一种新的块坐标下降方法OBCD，用于解决具有正交约束的一般非光滑组合问题。 OBCD是一种可行的方法，具有低的计算复杂性，并且获得严格的收敛保证。

    

    具有正交约束的非光滑组合优化在统计学习和数据科学中有广泛的应用。由于其非凸性和非光滑性质，该问题通常很难求解。现有的解决方案受到以下一个或多个限制的限制：（i）它们是需要每次迭代高计算成本的全梯度方法；（ii）它们无法解决一般的非光滑组合问题；（iii）它们是不可行方法，并且只能在极限点处实现解的可行性；（iv）它们缺乏严格的收敛保证；（v）它们只能获得关键点的弱最优性。在本文中，我们提出了一种新的块坐标下降方法OBCD，用于解决正交约束下的一般非光滑组合问题。OBCD是一种可行的方法，具有低的计算复杂性。在每次迭代中，我们的算法会更新...

    Nonsmooth composite optimization with orthogonality constraints has a broad spectrum of applications in statistical learning and data science. However, this problem is generally challenging to solve due to its non-convex and non-smooth nature. Existing solutions are limited by one or more of the following restrictions: (i) they are full gradient methods that require high computational costs in each iteration; (ii) they are not capable of solving general nonsmooth composite problems; (iii) they are infeasible methods and can only achieve the feasibility of the solution at the limit point; (iv) they lack rigorous convergence guarantees; (v) they only obtain weak optimality of critical points. In this paper, we propose \textit{\textbf{OBCD}}, a new Block Coordinate Descent method for solving general nonsmooth composite problems under Orthogonality constraints. \textit{\textbf{OBCD}} is a feasible method with low computation complexity footprints. In each iteration, our algorithm updates $
    

