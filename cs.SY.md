# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning-based Receding Horizon Control using Adaptive Control Barrier Functions for Safety-Critical Systems](https://arxiv.org/abs/2403.17338) | 提出了一种基于强化学习的滚动视野控制方法，利用自适应控制屏障函数，以解决安全关键系统中性能和可行性受影响的问题 |
| [^2] | [Sigma-point Kalman Filter with Nonlinear Unknown Input Estimation via Optimization and Data-driven Approach for Dynamic Systems.](http://arxiv.org/abs/2306.12361) | 本文提出了一种不需要假设未知输入为线性的方法，结合非线性优化和数据驱动方法可以实现对未知输入的估计，并通过联合 sigma-point 变换方案将状态和未知输入的不确定性纳入估计中，确保其稳定性。这个方法适用于许多智能自主系统。 |

# 详细

[^1]: 使用自适应控制屏障函数的基于强化学习的滚动视野控制用于安全关键系统

    Reinforcement Learning-based Receding Horizon Control using Adaptive Control Barrier Functions for Safety-Critical Systems

    [https://arxiv.org/abs/2403.17338](https://arxiv.org/abs/2403.17338)

    提出了一种基于强化学习的滚动视野控制方法，利用自适应控制屏障函数，以解决安全关键系统中性能和可行性受影响的问题

    

    最优控制方法为安全关键问题提供解决方案，但很容易变得棘手。控制屏障函数(CBFs)作为一种流行技术出现，通过其前向不变性属性，有利于通过在损失一些性能的情况下，显式地保证安全。该方法涉及定义性能目标以及必须始终执行的基于CBF的安全约束。遗憾的是，两个关键因素可能会对性能和解决方案的可行性产生显著影响：(i)成本函数及其相关参数的选择，以及(ii)在CBF约束内进行参数校准，捕捉性能和保守性之间的折衷，以及不可行性。为了解决这些挑战，我们提出了一种利用模型预测控制(MPC)的强化学习(RL)滚动视野控制(RHC)方法。

    arXiv:2403.17338v1 Announce Type: cross  Abstract: Optimal control methods provide solutions to safety-critical problems but easily become intractable. Control Barrier Functions (CBFs) have emerged as a popular technique that facilitates their solution by provably guaranteeing safety, through their forward invariance property, at the expense of some performance loss. This approach involves defining a performance objective alongside CBF-based safety constraints that must always be enforced. Unfortunately, both performance and solution feasibility can be significantly impacted by two key factors: (i) the selection of the cost function and associated parameters, and (ii) the calibration of parameters within the CBF-based constraints, which capture the trade-off between performance and conservativeness. %as well as infeasibility. To address these challenges, we propose a Reinforcement Learning (RL)-based Receding Horizon Control (RHC) approach leveraging Model Predictive Control (MPC) with
    
[^2]: 基于优化和数据驱动的 sigma-point 卡尔曼滤波器与非线性未知输入估计器

    Sigma-point Kalman Filter with Nonlinear Unknown Input Estimation via Optimization and Data-driven Approach for Dynamic Systems. (arXiv:2306.12361v1 [eess.SY])

    [http://arxiv.org/abs/2306.12361](http://arxiv.org/abs/2306.12361)

    本文提出了一种不需要假设未知输入为线性的方法，结合非线性优化和数据驱动方法可以实现对未知输入的估计，并通过联合 sigma-point 变换方案将状态和未知输入的不确定性纳入估计中，确保其稳定性。这个方法适用于许多智能自主系统。

    

    多数关于状态和未知输入(UI)估计的文献都要求UI是线性的，这个限制可能太严格了，因为它并不适用于许多智能自主系统。为了克服这一限制，我们提出了一种无导数未知输入 Sigma-point 卡尔曼滤波器(SPKE-nUI)，其中 SPKF 与普通非线性 UI 估计器相互连接，可以通过非线性优化和数据驱动方法实现。非线性 UI 估计器使用后验状态估计，这对状态预测误差不太敏感。此外，我们引入了联合 sigma-point 变换方案，将状态和 UI 的不确定性纳入 SPKF-nUI 的估计中。深入的随机稳定性分析证明了在合理的假设下，所提出的 SPKF-nUI 可以产生指数级收敛的估计误差界限。最后，我们在基于模拟的路面车辆控制问题上进行了两个案例研究。

    Most works on joint state and unknown input (UI) estimation require the assumption that the UIs are linear; this is potentially restrictive as it does not hold in many intelligent autonomous systems. To overcome this restriction and circumvent the need to linearize the system, we propose a derivative-free Unknown Input Sigma-point Kalman Filter (SPKF-nUI) where the SPKF is interconnected with a general nonlinear UI estimator that can be implemented via nonlinear optimization and data-driven approaches. The nonlinear UI estimator uses the posterior state estimate which is less susceptible to state prediction error. In addition, we introduce a joint sigma-point transformation scheme to incorporate both the state and UI uncertainties in the estimation of SPKF-nUI. An in-depth stochastic stability analysis proves that the proposed SPKF-nUI yields exponentially converging estimation error bounds under reasonable assumptions. Finally, two case studies are carried out on a simulation-based ri
    

