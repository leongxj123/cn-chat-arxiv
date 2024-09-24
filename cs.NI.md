# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ForestColl: Efficient Collective Communications on Heterogeneous Network Fabrics](https://arxiv.org/abs/2402.06787) | ForestColl是一种针对任意网络拓扑生成高效调度的工具，通过构建广播/聚合生成跨越树的通信调度，实现了理论上的最小网络拥塞，并在实验中表现出高于供应商自带通信库的性能。 |
| [^2] | [FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations](https://arxiv.org/abs/2209.14399) | 提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。 |

# 详细

[^1]: ForestColl: 异构网络结构上高效的集合通信

    ForestColl: Efficient Collective Communications on Heterogeneous Network Fabrics

    [https://arxiv.org/abs/2402.06787](https://arxiv.org/abs/2402.06787)

    ForestColl是一种针对任意网络拓扑生成高效调度的工具，通过构建广播/聚合生成跨越树的通信调度，实现了理论上的最小网络拥塞，并在实验中表现出高于供应商自带通信库的性能。

    

    随着现代深度神经网络模型越来越大，加速器之间的集合通信（如allreduce等）成为一个重要的性能瓶颈。在当今高度多样化和异构的网络结构下设计高效的通信调度是一项具有挑战性的任务。本文提出了一种名为ForestColl的工具，它能够为任意网络拓扑生成高效的调度。ForestColl使用广播/聚合生成跨越树作为通信调度，实现了理论上的最小网络拥塞。其调度生成运行在强多项式时间内，且具有高扩展性。ForestColl支持包括交换网络和直接连接在内的任何网络结构，以及任何网络图结构。我们在多集群的AMD MI250和NVIDIA A100平台上评估了ForestColl。与供应商自己优化的通信库RCCL和NCCL相比，ForestColl的调度性能提高了高达52％。ForestColl还优于其他...

    As modern DNN models grow ever larger, collective communications between the accelerators (allreduce, etc.) emerge as a significant performance bottleneck. Designing efficient communication schedules is challenging given today's highly diverse and heterogeneous network fabrics. In this paper, we present ForestColl, a tool that generates efficient schedules for any network topology. ForestColl constructs broadcast/aggregation spanning trees as the communication schedule, achieving theoretically minimum network congestion. Its schedule generation runs in strongly polynomial time and is highly scalable. ForestColl supports any network fabrics, including both switching fabrics and direct connections, as well as any network graph structure. We evaluated ForestColl on multi-cluster AMD MI250 and NVIDIA A100 platforms. ForestColl's schedules achieved up to 52\% higher performance compared to the vendors' own optimized communication libraries, RCCL and NCCL. ForestColl also outperforms other s
    
[^2]: FIRE：面向边缘计算迁移的故障自适应强化学习框架

    FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations

    [https://arxiv.org/abs/2209.14399](https://arxiv.org/abs/2209.14399)

    提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。

    

    在边缘计算中，用户服务配置文件由于用户移动而进行迁移。已经提出了强化学习（RL）框架来进行迁移，通常是在模拟数据上进行训练。然而，现有的RL框架忽视了偶发的服务器故障，尽管罕见，但会影响到像自动驾驶和实时障碍检测等对延迟敏感的应用。因此，这些（罕见事件）故障虽然在历史训练数据中没有得到充分代表，却对基于数据驱动的RL算法构成挑战。由于在实际应用中调整故障频率进行训练是不切实际的，我们引入了FIRE，这是一个通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件的框架。我们提出了ImRE，一种基于重要性抽样的Q-learning算法，它根据罕见事件对值函数的影响进行比例抽样。FIRE考虑了延迟、迁移、故障和备份pl

    arXiv:2209.14399v2 Announce Type: replace-cross  Abstract: In edge computing, users' service profiles are migrated due to user mobility. Reinforcement learning (RL) frameworks have been proposed to do so, often trained on simulated data. However, existing RL frameworks overlook occasional server failures, which although rare, impact latency-sensitive applications like autonomous driving and real-time obstacle detection. Nevertheless, these failures (rare events), being not adequately represented in historical training data, pose a challenge for data-driven RL algorithms. As it is impractical to adjust failure frequency in real-world applications for training, we introduce FIRE, a framework that adapts to rare events by training a RL policy in an edge computing digital twin environment. We propose ImRE, an importance sampling-based Q-learning algorithm, which samples rare events proportionally to their impact on the value function. FIRE considers delay, migration, failure, and backup pl
    

