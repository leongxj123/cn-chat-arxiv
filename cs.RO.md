# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling Is All You Need: Autonomous Driving with JAX-Accelerated Reinforcement Learning](https://arxiv.org/abs/2312.15122) | 本研究提出了一种扩展的自动驾驶强化学习方法，在大规模实验中展示了随着规模增加，策略性能的改善。与现有机器学习自动驾驶策略相比，我们的最佳策略将故障率降低了64％，同时提高了25％的驾驶进展速度。 |

# 详细

[^1]: 扩展就是一切：使用JAX加速强化学习的自动驾驶

    Scaling Is All You Need: Autonomous Driving with JAX-Accelerated Reinforcement Learning

    [https://arxiv.org/abs/2312.15122](https://arxiv.org/abs/2312.15122)

    本研究提出了一种扩展的自动驾驶强化学习方法，在大规模实验中展示了随着规模增加，策略性能的改善。与现有机器学习自动驾驶策略相比，我们的最佳策略将故障率降低了64％，同时提高了25％的驾驶进展速度。

    

    强化学习已经在复杂领域如视频游戏中展现出超越最优人类的能力。然而，为自动驾驶运行必要规模的强化学习实验非常困难。构建一个大规模的强化学习系统并在多个GPU上进行分布是具有挑战性的。在训练过程中在真实世界车辆上收集经验从安全和可扩展性的角度来看是不可行的。因此，需要一个高效且真实的驾驶模拟器，使用大量来自真实驾驶的数据。我们将这些能力集合在一起，并进行大规模的强化学习实验用于自动驾驶。我们证明，随着规模的增加，我们的策略表现得到了提升。我们最佳策略将故障率降低了64％，同时比现有机器学习自动驾驶策略提高了25％的驾驶进展速度。

    Reinforcement learning has been demonstrated to outperform even the best humans in complex domains like video games. However, running reinforcement learning experiments on the required scale for autonomous driving is extremely difficult. Building a large scale reinforcement learning system and distributing it across many GPUs is challenging. Gathering experience during training on real world vehicles is prohibitive from a safety and scalability perspective. Therefore, an efficient and realistic driving simulator is required that uses a large amount of data from real-world driving. We bring these capabilities together and conduct large-scale reinforcement learning experiments for autonomous driving. We demonstrate that our policy performance improves with increasing scale. Our best performing policy reduces the failure rate by 64% while improving the rate of driving progress by 25% compared to the policies produced by state-of-the-art machine learning for autonomous driving.
    

