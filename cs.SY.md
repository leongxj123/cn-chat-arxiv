# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Inverse Learning in Stackelberg Trajectory Games.](http://arxiv.org/abs/2308.08017) | 这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。 |
| [^2] | [RL + Model-based Control: Using On-demand Optimal Control to Learn Versatile Legged Locomotion.](http://arxiv.org/abs/2305.17842) | 本文提出了一种 RL+模型控制框架以开发出可以有效可靠地学习的健壮控制策略，通过整合有限时间最优控制生成的按需参考运动分散 RL 过程，同时克服了建模简化的固有局限性，在足式 locomotion 上实现了多功能和强健，能泛化参考运动并处理更复杂的运动任务。 |

# 详细

[^1]: Stackelberg轨迹博弈中的主动逆向学习

    Active Inverse Learning in Stackelberg Trajectory Games. (arXiv:2308.08017v1 [cs.GT])

    [http://arxiv.org/abs/2308.08017](http://arxiv.org/abs/2308.08017)

    这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。

    

    博弈论的逆向学习是从玩家的行为中推断出他们的目标的问题。我们在一个Stackelberg博弈中，通过每个玩家的动态系统轨迹来定义一个逆向学习问题，其中包括一个领导者和一个跟随者。我们提出了一种主动逆向学习方法，让领导者推断出一个有限候选集中描述跟随者目标函数的假设。与现有方法使用被动观察到的轨迹不同，所提出的方法主动地最大化不同假设下跟随者轨迹的差异，加速领导者的推断过程。我们在一个递进的重复轨迹博弈中展示了所提出的方法。与均匀随机输入相比，所提供的方法加速了概率收敛到条件于跟随者轨迹的不同假设上的收敛速度。

    Game-theoretic inverse learning is the problem of inferring the players' objectives from their actions. We formulate an inverse learning problem in a Stackelberg game between a leader and a follower, where each player's action is the trajectory of a dynamical system. We propose an active inverse learning method for the leader to infer which hypothesis among a finite set of candidates describes the follower's objective function. Instead of using passively observed trajectories like existing methods, the proposed method actively maximizes the differences in the follower's trajectories under different hypotheses to accelerate the leader's inference. We demonstrate the proposed method in a receding-horizon repeated trajectory game. Compared with uniformly random inputs, the leader inputs provided by the proposed method accelerate the convergence of the probability of different hypotheses conditioned on the follower's trajectory by orders of magnitude.
    
[^2]: RL+模型控制：使用按需最优控制学习多功能足式 locomotion

    RL + Model-based Control: Using On-demand Optimal Control to Learn Versatile Legged Locomotion. (arXiv:2305.17842v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2305.17842](http://arxiv.org/abs/2305.17842)

    本文提出了一种 RL+模型控制框架以开发出可以有效可靠地学习的健壮控制策略，通过整合有限时间最优控制生成的按需参考运动分散 RL 过程，同时克服了建模简化的固有局限性，在足式 locomotion 上实现了多功能和强健，能泛化参考运动并处理更复杂的运动任务。

    

    本文提出了一种控制框架，将基于模型的最优控制和强化学习（RL）相结合，实现了多功能和强健的足式 locomotion。我们的方法通过整合有限时间最优控制生成的按需参考运动来增强 RL 训练过程，覆盖了广泛的速度和步态。这些参考运动作为 RL 策略模仿的目标，导致开发出可有效可靠地学习的健壮控制策略。此外，通过考虑全身动力学，RL 克服了建模简化的固有局限性。通过仿真和硬件实验，我们展示了 RL 训练过程在我们的框架内的强健性和可控性。此外，我们的方法展示了泛化参考运动和处理可能对简化模型构成挑战的更复杂的运动任务的能力，利用了 RL 的灵活性。

    This letter presents a control framework that combines model-based optimal control and reinforcement learning (RL) to achieve versatile and robust legged locomotion. Our approach enhances the RL training process by incorporating on-demand reference motions generated through finite-horizon optimal control, covering a broad range of velocities and gaits. These reference motions serve as targets for the RL policy to imitate, resulting in the development of robust control policies that can be learned efficiently and reliably. Moreover, by considering whole-body dynamics, RL overcomes the inherent limitations of modelling simplifications. Through simulation and hardware experiments, we demonstrate the robustness and controllability of the RL training process within our framework. Furthermore, our method demonstrates the ability to generalize reference motions and handle more complex locomotion tasks that may pose challenges for the simplified model, leveraging the flexibility of RL.
    

