# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method](https://arxiv.org/abs/2403.02811) | 本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。 |
| [^2] | [On Building Myopic MPC Policies using Supervised Learning.](http://arxiv.org/abs/2401.12546) | 本论文提出了一种使用监督学习构建近视MPC策略的方法，通过离线学习最优值函数，可以显著减少在线计算负担，而不影响控制器的性能。 |

# 详细

[^1]: 具有Koopman算子学习和Nyström方法的非线性系统的线性二次控制

    Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method

    [https://arxiv.org/abs/2403.02811](https://arxiv.org/abs/2403.02811)

    本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。

    

    在本文中，我们研究了Koopman算子框架如何与核方法相结合以有效控制非线性动力系统。虽然核方法通常具有很大的计算需求，但我们展示了随机子空间（Nyström逼近）如何实现巨大的计算节约，同时保持精度。我们的主要技术贡献在于推导出关于Nyström逼近效果的理论保证。更具体地说，我们研究了线性二次调节器问题，证明了对于最优控制问题的相关解的近似Riccati算子和调节器目标都以$ m^{-1/2} $的速率收敛，其中$ m $是随机子空间大小。理论发现得到了数值实验的支持。

    arXiv:2403.02811v1 Announce Type: cross  Abstract: In this paper, we study how the Koopman operator framework can be combined with kernel methods to effectively control nonlinear dynamical systems. While kernel methods have typically large computational requirements, we show how random subspaces (Nystr\"om approximation) can be used to achieve huge computational savings while preserving accuracy. Our main technical contribution is deriving theoretical guarantees on the effect of the Nystr\"om approximation. More precisely, we study the linear quadratic regulator problem, showing that both the approximated Riccati operator and the regulator objective, for the associated solution of the optimal control problem, converge at the rate $m^{-1/2}$, where $m$ is the random subspace size. Theoretical findings are complemented by numerical experiments corroborating our results.
    
[^2]: 使用监督学习构建近视MPC策略

    On Building Myopic MPC Policies using Supervised Learning. (arXiv:2401.12546v1 [cs.LG])

    [http://arxiv.org/abs/2401.12546](http://arxiv.org/abs/2401.12546)

    本论文提出了一种使用监督学习构建近视MPC策略的方法，通过离线学习最优值函数，可以显著减少在线计算负担，而不影响控制器的性能。

    

    近期，在模型预测控制（MPC）中，结合监督学习技术应用引起了广泛关注，尤其是在近似显式MPC领域，其中使用函数逼近器（如深度神经网络）通过离线生成的最佳状态-动作对来学习MPC策略。虽然近似显式MPC的目标是尽可能准确地复制MPC策略，用训练好的神经网络替代在线优化，但通常会失去解决在线优化问题所带来的性能保证。本文提出了一种替代策略，即使用监督学习离线学习最优值函数而不是学习最优策略。然后，在一个非常短的预测时间范围内，将其作为近视MPC的成本函数，从而显著减少在线计算负担，不影响控制器的性能。该方法与现有方法存在差异。

    The application of supervised learning techniques in combination with model predictive control (MPC) has recently generated significant interest, particularly in the area of approximate explicit MPC, where function approximators like deep neural networks are used to learn the MPC policy via optimal state-action pairs generated offline. While the aim of approximate explicit MPC is to closely replicate the MPC policy, substituting online optimization with a trained neural network, the performance guarantees that come with solving the online optimization problem are typically lost. This paper considers an alternative strategy, where supervised learning is used to learn the optimal value function offline instead of learning the optimal policy. This can then be used as the cost-to-go function in a myopic MPC with a very short prediction horizon, such that the online computation burden reduces significantly without affecting the controller performance. This approach differs from existing wor
    

