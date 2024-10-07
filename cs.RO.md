# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Artificial consciousness. Some logical and conceptual preliminaries](https://arxiv.org/abs/2403.20177) | 需要在人工系统中平衡讨论意识的可能实现，提出了使用意识的维度和特征来进行讨论的必要性。 |
| [^2] | [Efficient and Guaranteed-Safe Non-Convex Trajectory Optimization with Constrained Diffusion Model](https://arxiv.org/abs/2403.05571) | 本文提出了一种具有约束扩散模型的高效和保证安全的非凸轨迹优化框架，通过结合扩散模型和数值求解器，保证了计算效率和约束满足。 |
| [^3] | [Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning.](http://arxiv.org/abs/2401.08632) | 将质量多样性优化与描述符条件加强学习相结合，以克服进化算法的局限性，并在生成既多样又高性能的解决方案集合方面取得成功。 |
| [^4] | [Imitation Learning from Observation through Optimal Transport.](http://arxiv.org/abs/2310.01632) | 本文提出了一种通过最优输运进行从观察中的模仿学习的方法，该方法不需要学习模型或对抗学习，可以与任何强化学习算法集成，并在各种连续控制任务上超过了现有最先进方法，在ILfO设置下实现了专家级的性能。 |

# 详细

[^1]: 人工意识。一些逻辑和概念初步

    Artificial consciousness. Some logical and conceptual preliminaries

    [https://arxiv.org/abs/2403.20177](https://arxiv.org/abs/2403.20177)

    需要在人工系统中平衡讨论意识的可能实现，提出了使用意识的维度和特征来进行讨论的必要性。

    

    arXiv:2403.20177v1 公告类型: 新的 摘要: 人工意识在理论上是否可能？是否合乎情理？如果是，那么技术上可行吗？要解决这些问题，有必要奠定一些基础，阐明人工意识产生的逻辑和经验条件以及涉及的相关术语的含义。意识是一个多义词：来自不同领域的研究人员，包括神经科学、人工智能、机器人技术和哲学等，有时会使用不同术语来指称相同现象，或者使用相同术语来指称不同现象。事实上，如果我们想探讨人工意识，就需要恰当界定关键概念。在此，经过一些逻辑和概念初步工作后，我们认为有必要使用意识的维度和特征进行平衡讨论，探讨它们在人工系统中的可能实例化或实现。我们在这项工作的主要目标是...

    arXiv:2403.20177v1 Announce Type: new  Abstract: Is artificial consciousness theoretically possible? Is it plausible? If so, is it technically feasible? To make progress on these questions, it is necessary to lay some groundwork clarifying the logical and empirical conditions for artificial consciousness to arise and the meaning of relevant terms involved. Consciousness is a polysemic word: researchers from different fields, including neuroscience, Artificial Intelligence, robotics, and philosophy, among others, sometimes use different terms in order to refer to the same phenomena or the same terms to refer to different phenomena. In fact, if we want to pursue artificial consciousness, a proper definition of the key concepts is required. Here, after some logical and conceptual preliminaries, we argue for the necessity of using dimensions and profiles of consciousness for a balanced discussion about their possible instantiation or realisation in artificial systems. Our primary goal in t
    
[^2]: 具有约束扩散模型的高效和保证安全的非凸轨迹优化

    Efficient and Guaranteed-Safe Non-Convex Trajectory Optimization with Constrained Diffusion Model

    [https://arxiv.org/abs/2403.05571](https://arxiv.org/abs/2403.05571)

    本文提出了一种具有约束扩散模型的高效和保证安全的非凸轨迹优化框架，通过结合扩散模型和数值求解器，保证了计算效率和约束满足。

    

    机器人轨迹优化面临一个具有挑战性的非凸问题，这是由于复杂的动力学和环境设置造成的。本文引入了一个通用且完全可并行化的框架，将扩散模型和数值求解器结合起来，用于非凸轨迹优化，确保计算效率和约束满足。提出了一种新颖的带有额外约束违反损失的约束扩散模型进行训练。它旨在在采样过程中近似局部最优解的分布，同时最小化约束违反。然后用样本作为数值求解器的初始猜测，来优化并得出最终解，并验证可行性和最优性。

    arXiv:2403.05571v1 Announce Type: cross  Abstract: Trajectory optimization in robotics poses a challenging non-convex problem due to complex dynamics and environmental settings. Traditional numerical optimization methods are time-consuming in finding feasible solutions, whereas data-driven approaches lack safety guarantees for the output trajectories. In this paper, we introduce a general and fully parallelizable framework that combines diffusion models and numerical solvers for non-convex trajectory optimization, ensuring both computational efficiency and constraint satisfaction. A novel constrained diffusion model is proposed with an additional constraint violation loss for training. It aims to approximate the distribution of locally optimal solutions while minimizing constraint violations during sampling. The samples are then used as initial guesses for a numerical solver to refine and derive final solutions with formal verification of feasibility and optimality. Experimental evalua
    
[^3]: 将质量多样性与描述符条件加强学习相结合

    Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning. (arXiv:2401.08632v1 [cs.NE])

    [http://arxiv.org/abs/2401.08632](http://arxiv.org/abs/2401.08632)

    将质量多样性优化与描述符条件加强学习相结合，以克服进化算法的局限性，并在生成既多样又高性能的解决方案集合方面取得成功。

    

    智能的基本特征之一是找到新颖和有创造性的解决方案来解决给定的挑战或适应未预料到的情况。质量多样性优化是一类进化算法，可以生成既多样又高性能的解决方案集合。其中，MAP-Elites是一个著名的例子，已成功应用于各种领域，包括进化机器人学。然而，MAP-Elites通过遗传算法的随机突变进行发散搜索，因此仅限于进化低维解决方案的种群。PGA-MAP-Elites通过受深度强化学习启发的基于梯度的变异算子克服了这一限制，从而实现了大型神经网络的进化。尽管在许多环境中性能优秀，但PGA-MAP-Elites在一些任务中失败，其中基于梯度的变异算子的收敛搜索阻碍了多样性。在这项工作中，我们...

    A fundamental trait of intelligence involves finding novel and creative solutions to address a given challenge or to adapt to unforeseen situations. Reflecting this, Quality-Diversity optimization is a family of Evolutionary Algorithms, that generates collections of both diverse and high-performing solutions. Among these, MAP-Elites is a prominent example, that has been successfully applied to a variety of domains, including evolutionary robotics. However, MAP-Elites performs a divergent search with random mutations originating from Genetic Algorithms, and thus, is limited to evolving populations of low-dimensional solutions. PGA-MAP-Elites overcomes this limitation using a gradient-based variation operator inspired by deep reinforcement learning which enables the evolution of large neural networks. Although high-performing in many environments, PGA-MAP-Elites fails on several tasks where the convergent search of the gradient-based variation operator hinders diversity. In this work, we
    
[^4]: 通过最优输运进行从观察中的模仿学习

    Imitation Learning from Observation through Optimal Transport. (arXiv:2310.01632v1 [cs.RO])

    [http://arxiv.org/abs/2310.01632](http://arxiv.org/abs/2310.01632)

    本文提出了一种通过最优输运进行从观察中的模仿学习的方法，该方法不需要学习模型或对抗学习，可以与任何强化学习算法集成，并在各种连续控制任务上超过了现有最先进方法，在ILfO设置下实现了专家级的性能。

    

    从观察中的模仿学习（ILfO）是一种学习者试图在没有直接指导的情况下，使用观测数据模仿专家行为的设置。在本文中，我们重新审视了最优输运在IL中的应用，其中根据学习者和专家的状态轨迹之间的Wasserstein距离生成奖励。我们证明了现有方法可以简化为生成无需学习模型或对抗学习的奖励函数。与许多其他最先进的方法不同，我们的方法可以与任何强化学习算法集成，并适用于ILfO。我们在各种连续控制任务上展示了这种简单方法的有效性，并发现即使只观察单个专家轨迹而没有动作，它在ILfO设置中超过了现有最先进方法，在一系列评估领域中实现了专家级的性能。

    Imitation Learning from Observation (ILfO) is a setting in which a learner tries to imitate the behavior of an expert, using only observational data and without the direct guidance of demonstrated actions. In this paper, we re-examine the use of optimal transport for IL, in which a reward is generated based on the Wasserstein distance between the state trajectories of the learner and expert. We show that existing methods can be simplified to generate a reward function without requiring learned models or adversarial learning. Unlike many other state-of-the-art methods, our approach can be integrated with any RL algorithm, and is amenable to ILfO. We demonstrate the effectiveness of this simple approach on a variety of continuous control tasks and find that it surpasses the state of the art in the IlfO setting, achieving expert-level performance across a range of evaluation domains even when observing only a single expert trajectory without actions.
    

