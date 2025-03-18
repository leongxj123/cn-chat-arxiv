# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization in Mixed Discrete-Continuous MDPs.](http://arxiv.org/abs/2401.12243) | Constraint-Generation Policy Optimization (CGPO)是一种针对混合离散连续MDPs的策略优化方法，能够提供有界的策略误差保证，推导出最优策略，并生成最坏情况的状态轨迹来诊断策略缺陷。 |

# 详细

[^1]: Constraint-Generation Policy Optimization (CGPO): 针对混合离散连续MDPs中的策略优化的非线性规划

    Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization in Mixed Discrete-Continuous MDPs. (arXiv:2401.12243v1 [math.OC])

    [http://arxiv.org/abs/2401.12243](http://arxiv.org/abs/2401.12243)

    Constraint-Generation Policy Optimization (CGPO)是一种针对混合离散连续MDPs的策略优化方法，能够提供有界的策略误差保证，推导出最优策略，并生成最坏情况的状态轨迹来诊断策略缺陷。

    

    我们提出了Constraint-Generation Policy Optimization (CGPO)方法，用于在混合离散连续Markov Decision Processes (DC-MDPs)中优化策略参数。CGPO不仅能够提供有界的策略误差保证，覆盖具有表达能力的非线性动力学的无数初始状态范围的DC-MDPs，而且在结束时可以明确地推导出最优策略。此外，CGPO还能够生成最坏情况的状态轨迹来诊断策略缺陷，并提供最优行动的反事实解释。为了实现这些结果，CGPO提出了一个双层的混合整数非线性优化框架，用于在定义的表达能力类别（即分段(非)线性）内优化策略，并将其转化为一个最优的约束生成方法，通过对抗性生成最坏情况的状态轨迹。此外，借助现代非线性优化器，CGPO可以获得解决方案。

    We propose Constraint-Generation Policy Optimization (CGPO) for optimizing policy parameters within compact and interpretable policy classes for mixed discrete-continuous Markov Decision Processes (DC-MDPs). CGPO is not only able to provide bounded policy error guarantees over an infinite range of initial states for many DC-MDPs with expressive nonlinear dynamics, but it can also provably derive optimal policies in cases where it terminates with zero error. Furthermore, CGPO can generate worst-case state trajectories to diagnose policy deficiencies and provide counterfactual explanations of optimal actions. To achieve such results, CGPO proposes a bi-level mixed-integer nonlinear optimization framework for optimizing policies within defined expressivity classes (i.e. piecewise (non)-linear) and reduces it to an optimal constraint generation methodology that adversarially generates worst-case state trajectories. Furthermore, leveraging modern nonlinear optimizers, CGPO can obtain soluti
    

