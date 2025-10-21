# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations](https://arxiv.org/abs/2209.14399) | 提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。 |

# 详细

[^1]: FIRE：面向边缘计算迁移的故障自适应强化学习框架

    FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations

    [https://arxiv.org/abs/2209.14399](https://arxiv.org/abs/2209.14399)

    提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。

    

    在边缘计算中，用户服务配置文件由于用户移动而进行迁移。已经提出了强化学习（RL）框架来进行迁移，通常是在模拟数据上进行训练。然而，现有的RL框架忽视了偶发的服务器故障，尽管罕见，但会影响到像自动驾驶和实时障碍检测等对延迟敏感的应用。因此，这些（罕见事件）故障虽然在历史训练数据中没有得到充分代表，却对基于数据驱动的RL算法构成挑战。由于在实际应用中调整故障频率进行训练是不切实际的，我们引入了FIRE，这是一个通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件的框架。我们提出了ImRE，一种基于重要性抽样的Q-learning算法，它根据罕见事件对值函数的影响进行比例抽样。FIRE考虑了延迟、迁移、故障和备份pl

    arXiv:2209.14399v2 Announce Type: replace-cross  Abstract: In edge computing, users' service profiles are migrated due to user mobility. Reinforcement learning (RL) frameworks have been proposed to do so, often trained on simulated data. However, existing RL frameworks overlook occasional server failures, which although rare, impact latency-sensitive applications like autonomous driving and real-time obstacle detection. Nevertheless, these failures (rare events), being not adequately represented in historical training data, pose a challenge for data-driven RL algorithms. As it is impractical to adjust failure frequency in real-world applications for training, we introduce FIRE, a framework that adapts to rare events by training a RL policy in an edge computing digital twin environment. We propose ImRE, an importance sampling-based Q-learning algorithm, which samples rare events proportionally to their impact on the value function. FIRE considers delay, migration, failure, and backup pl
    

