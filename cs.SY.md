# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving a Proportional Integral Controller with Reinforcement Learning on a Throttle Valve Benchmark](https://arxiv.org/abs/2402.13654) | 通过引入强化学习与引导，结合比例积分（PI）控制器，本文提出了一种学习基础的控制策略，用于非线性节流阀的控制，实现了一个几乎最优的控制器。 |
| [^2] | [Herd Behavior in Optimal Investment: A Dual-Agent Approach with Investment Opinion and Rational Decision Decomposition.](http://arxiv.org/abs/2401.07183) | 本文研究了涉及双代理的最优投资问题，引入了平均偏差来衡量代理决策的差异程度。通过理性决策分解，分析了群体行为对最优决策的影响，并通过模拟验证了研究的有效性。 |
| [^3] | [Constrained Reinforcement Learning using Distributional Representation for Trustworthy Quadrotor UAV Tracking Control.](http://arxiv.org/abs/2302.11694) | 提出了一种使用分布式表示进行受限强化学习的方法，用于可信四旋翼无人机的跟踪控制。通过集成分布式强化学习干扰估计器和随机模型预测控制器，能够准确识别气动效应的不确定性，实现最优的全局收敛速率和一定的亚线性收敛速率。 |

# 详细

[^1]: 使用强化学习改进比例积分控制器在节流阀基准上的应用

    Improving a Proportional Integral Controller with Reinforcement Learning on a Throttle Valve Benchmark

    [https://arxiv.org/abs/2402.13654](https://arxiv.org/abs/2402.13654)

    通过引入强化学习与引导，结合比例积分（PI）控制器，本文提出了一种学习基础的控制策略，用于非线性节流阀的控制，实现了一个几乎最优的控制器。

    

    本文提出了一种基于学习的控制策略，用于非线性节流阀，该节流阀具有不对称的磁滞，实现了一个几乎最优的控制器，而不需要任何关于环境的先验知识。我们首先通过精心调整的比例积分（PI）控制器开始，并利用强化学习（RL）与引导的最新进展，通过从与阀门的额外交互中学习来改进闭环行为。我们在三个不同的阀门上的各种场景中测试了所提出的控制方法，所有这些都突显了将PI和RL框架结合以提高非线性随机系统控制性能的好处。在所有实验测试案例中，结果代理的样本效率都优于传统RL代理，并且优于PI控制器。

    arXiv:2402.13654v1 Announce Type: cross  Abstract: This paper presents a learning-based control strategy for non-linear throttle valves with an asymmetric hysteresis, leading to a near-optimal controller without requiring any prior knowledge about the environment. We start with a carefully tuned Proportional Integrator (PI) controller and exploit the recent advances in Reinforcement Learning (RL) with Guides to improve the closed-loop behavior by learning from the additional interactions with the valve. We test the proposed control method in various scenarios on three different valves, all highlighting the benefits of combining both PI and RL frameworks to improve control performance in non-linear stochastic systems. In all the experimental test cases, the resulting agent has a better sample efficiency than traditional RL agents and outperforms the PI controller.
    
[^2]: 最优投资中的群体行为: 带有投资意见和理性决策分解的双代理模型

    Herd Behavior in Optimal Investment: A Dual-Agent Approach with Investment Opinion and Rational Decision Decomposition. (arXiv:2401.07183v1 [eess.SY])

    [http://arxiv.org/abs/2401.07183](http://arxiv.org/abs/2401.07183)

    本文研究了涉及双代理的最优投资问题，引入了平均偏差来衡量代理决策的差异程度。通过理性决策分解，分析了群体行为对最优决策的影响，并通过模拟验证了研究的有效性。

    

    本文研究了涉及两个代理的最优投资问题，其中一个代理的决策受到另一个代理的影响。为了衡量两个代理决策之间的差异程度，我们引入了平均偏差。我们通过变分方法推导出了考虑群体行为的随机最优控制问题的解析解。我们从理论上分析了用户群体行为对最优决策的影响，并将其分解成理性决策，这被称为理性决策分解。此外，为了量化代理对自己的理性决策相对于另一个代理的偏好程度，我们引入了代理的投资意见。通过对真实股票数据的模拟验证了我们的研究。

    In this paper, we study the optimal investment problem involving two agents, where the decision of one agent is influenced by the other. To measure the distance between two agents' decisions, we introduce the average deviation. We formulate the stochastic optimal control problem considering herd behavior and derive the analytical solution through the variational method. We theoretically analyze the impact of users' herd behavior on the optimal decision by decomposing it into their rational decisions, which is called the rational decision decomposition. Furthermore, to quantify the preference for their rational decision over that of the other agent, we introduce the agent's investment opinion. Our study is validated through simulations on real stock data.
    
[^3]: 受限强化学习在可信四旋翼无人机跟踪控制中的应用

    Constrained Reinforcement Learning using Distributional Representation for Trustworthy Quadrotor UAV Tracking Control. (arXiv:2302.11694v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2302.11694](http://arxiv.org/abs/2302.11694)

    提出了一种使用分布式表示进行受限强化学习的方法，用于可信四旋翼无人机的跟踪控制。通过集成分布式强化学习干扰估计器和随机模型预测控制器，能够准确识别气动效应的不确定性，实现最优的全局收敛速率和一定的亚线性收敛速率。

    

    在复杂的动态环境中，同时实现四旋翼无人机的准确和可靠的跟踪控制是具有挑战性的。由于来自气动力的阻力和力矩变化是混沌的，并且难以精确识别，大多数现有的四旋翼跟踪系统将其视为传统控制方法中的简单“干扰”。本文提出了一种新颖的、可解释的轨迹跟踪器，将分布式强化学习干扰估计器与随机模型预测控制器（SMPC）相结合，用于未知的气动效应。所提出的估计器“受限分布式强化干扰估计器”（ConsDRED）准确地识别真实气动效应与估计值之间的不确定性。采用简化仿射干扰反馈进行控制参数化，以保证凸性，然后将其与SMPC相结合。我们在理论上保证ConsDRED至少实现最优的全局收敛速率和一定的亚线性收敛速率。

    Simultaneously accurate and reliable tracking control for quadrotors in complex dynamic environments is challenging. As aerodynamics derived from drag forces and moment variations are chaotic and difficult to precisely identify, most current quadrotor tracking systems treat them as simple `disturbances' in conventional control approaches. We propose a novel, interpretable trajectory tracker integrating a Distributional Reinforcement Learning disturbance estimator for unknown aerodynamic effects with a Stochastic Model Predictive Controller (SMPC). The proposed estimator `Constrained Distributional Reinforced disturbance estimator' (ConsDRED) accurately identifies uncertainties between true and estimated values of aerodynamic effects. Simplified Affine Disturbance Feedback is used for control parameterization to guarantee convexity, which we then integrate with a SMPC. We theoretically guarantee that ConsDRED achieves at least an optimal global convergence rate and a certain sublinear r
    

