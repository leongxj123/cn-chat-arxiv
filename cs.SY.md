# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning-based Receding Horizon Control using Adaptive Control Barrier Functions for Safety-Critical Systems](https://arxiv.org/abs/2403.17338) | 提出了一种基于强化学习的滚动视野控制方法，利用自适应控制屏障函数，以解决安全关键系统中性能和可行性受影响的问题 |

# 详细

[^1]: 使用自适应控制屏障函数的基于强化学习的滚动视野控制用于安全关键系统

    Reinforcement Learning-based Receding Horizon Control using Adaptive Control Barrier Functions for Safety-Critical Systems

    [https://arxiv.org/abs/2403.17338](https://arxiv.org/abs/2403.17338)

    提出了一种基于强化学习的滚动视野控制方法，利用自适应控制屏障函数，以解决安全关键系统中性能和可行性受影响的问题

    

    最优控制方法为安全关键问题提供解决方案，但很容易变得棘手。控制屏障函数(CBFs)作为一种流行技术出现，通过其前向不变性属性，有利于通过在损失一些性能的情况下，显式地保证安全。该方法涉及定义性能目标以及必须始终执行的基于CBF的安全约束。遗憾的是，两个关键因素可能会对性能和解决方案的可行性产生显著影响：(i)成本函数及其相关参数的选择，以及(ii)在CBF约束内进行参数校准，捕捉性能和保守性之间的折衷，以及不可行性。为了解决这些挑战，我们提出了一种利用模型预测控制(MPC)的强化学习(RL)滚动视野控制(RHC)方法。

    arXiv:2403.17338v1 Announce Type: cross  Abstract: Optimal control methods provide solutions to safety-critical problems but easily become intractable. Control Barrier Functions (CBFs) have emerged as a popular technique that facilitates their solution by provably guaranteeing safety, through their forward invariance property, at the expense of some performance loss. This approach involves defining a performance objective alongside CBF-based safety constraints that must always be enforced. Unfortunately, both performance and solution feasibility can be significantly impacted by two key factors: (i) the selection of the cost function and associated parameters, and (ii) the calibration of parameters within the CBF-based constraints, which capture the trade-off between performance and conservativeness. %as well as infeasibility. To address these challenges, we propose a Reinforcement Learning (RL)-based Receding Horizon Control (RHC) approach leveraging Model Predictive Control (MPC) with
    

