# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robotic Table Tennis: A Case Study into a High Speed Learning System.](http://arxiv.org/abs/2309.03315) | 本研究深入研究了一种真实世界的机器人学习系统，该系统能够与人类进行数百次的乒乓球回合，并且能够精确地将球返回到预定的目标位置。该系统整合了感知子系统、高速低延迟机器人控制器、仿真范例、自动重置真实世界环境等功能，并对系统的设计决策和重要性进行了详细描述。 |
| [^2] | [ArrayBot: Reinforcement Learning for Generalizable Distributed Manipulation through Touch.](http://arxiv.org/abs/2306.16857) | ArrayBot通过强化学习实现了通用分布式操作，通过对动作空间的重新定义和采用触觉观察训练，其控制策略不仅能够推广到未见过的物体形状，还能在实际机器人中进行转移，展示了巨大的潜力。 |

# 详细

[^1]: 机器人乒乓球：一个高速学习系统的案例研究

    Robotic Table Tennis: A Case Study into a High Speed Learning System. (arXiv:2309.03315v1 [cs.RO])

    [http://arxiv.org/abs/2309.03315](http://arxiv.org/abs/2309.03315)

    本研究深入研究了一种真实世界的机器人学习系统，该系统能够与人类进行数百次的乒乓球回合，并且能够精确地将球返回到预定的目标位置。该系统整合了感知子系统、高速低延迟机器人控制器、仿真范例、自动重置真实世界环境等功能，并对系统的设计决策和重要性进行了详细描述。

    

    我们展示了一个真实世界的机器人学习系统的深入研究，此前的工作已经表明该系统能够与人类进行数百次的乒乓球回合，并且能够精确地将球返回到预定的目标位置。该系统结合了高度优化的感知子系统、高速低延迟的机器人控制器、防止现实世界中损坏并能够进行零-shot转移策略训练的仿真范例，以及自动重置真实世界环境，使自主训练和评估在物理机器人上成为可能。我们通过详细描述整个系统，包括通常不广泛传播的大量设计决策，并结合一系列研究来阐明缓解各种延迟源的重要性、考虑训练和部署分布变化、感知系统的稳健性、策略超参数的敏感性和动作空间选择等方面的重要性。视频展示了系统的组件。

    We present a deep-dive into a real-world robotic learning system that, in previous work, was shown to be capable of hundreds of table tennis rallies with a human and has the ability to precisely return the ball to desired targets. This system puts together a highly optimized perception subsystem, a high-speed low-latency robot controller, a simulation paradigm that can prevent damage in the real world and also train policies for zero-shot transfer, and automated real world environment resets that enable autonomous training and evaluation on physical robots. We complement a complete system description, including numerous design decisions that are typically not widely disseminated, with a collection of studies that clarify the importance of mitigating various sources of latency, accounting for training and deployment distribution shifts, robustness of the perception system, sensitivity to policy hyper-parameters, and choice of action space. A video demonstrating the components of the sys
    
[^2]: ArrayBot: 通过触觉实现通用分布式操作的强化学习

    ArrayBot: Reinforcement Learning for Generalizable Distributed Manipulation through Touch. (arXiv:2306.16857v1 [cs.RO])

    [http://arxiv.org/abs/2306.16857](http://arxiv.org/abs/2306.16857)

    ArrayBot通过强化学习实现了通用分布式操作，通过对动作空间的重新定义和采用触觉观察训练，其控制策略不仅能够推广到未见过的物体形状，还能在实际机器人中进行转移，展示了巨大的潜力。

    

    我们介绍了ArrayBot，这是一个由16×16的竖向滑动柱和触觉传感器组成的分布式操作系统，可以同时支持、感知和操作桌面上的物体。为了实现通用分布式操作，我们利用强化学习算法自动发现控制策略。面对大量冗余的动作，我们提出考虑空间局部动作图块和频域中低频动作来重新定义动作空间。通过这个重新定义的动作空间，我们训练强化学习代理，只通过触觉观察即可重新定位不同的物体。令人惊讶的是，我们发现发现的策略不仅可以推广到模拟器中看不见的物体形状，而且可以在物理机器人上进行转移而不需要任何域随机化。利用部署的策略，我们展示了丰富的真实世界操作任务，展示了其巨大潜力。

    We present ArrayBot, a distributed manipulation system consisting of a $16 \times 16$ array of vertically sliding pillars integrated with tactile sensors, which can simultaneously support, perceive, and manipulate the tabletop objects. Towards generalizable distributed manipulation, we leverage reinforcement learning (RL) algorithms for the automatic discovery of control policies. In the face of the massively redundant actions, we propose to reshape the action space by considering the spatially local action patch and the low-frequency actions in the frequency domain. With this reshaped action space, we train RL agents that can relocate diverse objects through tactile observations only. Surprisingly, we find that the discovered policy can not only generalize to unseen object shapes in the simulator but also transfer to the physical robot without any domain randomization. Leveraging the deployed policy, we present abundant real-world manipulation tasks, illustrating the vast potential of
    

