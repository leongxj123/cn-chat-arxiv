# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421) | DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。 |

# 详细

[^1]: DiffTOP: 可微分轨迹优化在强化学习和模仿学习中的应用

    DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2402.05421](https://arxiv.org/abs/2402.05421)

    DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。

    

    本文介绍了DiffTOP，它利用可微分轨迹优化作为策略表示，为深度强化学习和模仿学习生成动作。轨迹优化是一种在控制领域中广泛使用的算法，由成本和动力学函数参数化。我们的方法的关键是利用了最近在可微分轨迹优化方面的进展，使得可以计算损失对于轨迹优化的参数的梯度。因此，轨迹优化的成本和动力学函数可以端到端地学习。DiffTOP解决了之前模型基于强化学习算法中的“目标不匹配”问题，因为DiffTOP中的动力学模型通过轨迹优化过程中的策略梯度损失直接最大化任务性能。我们还对DiffTOP在标准机器人操纵任务套件中进行了模仿学习性能基准测试。

    This paper introduces DiffTOP, which utilizes Differentiable Trajectory OPtimization as the policy representation to generate actions for deep reinforcement and imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTOP addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTOP is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTOP for imitation learning on standard robotic manipulation task suites with high-dimensional sensory
    

