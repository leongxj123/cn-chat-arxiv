# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Deep Recurrent-Reinforcement Learning Method for Intelligent AutoScaling of Serverless Functions.](http://arxiv.org/abs/2308.05937) | 该论文介绍了一种用于智能自动缩放无服务器函数的深度循环强化学习方法，针对波动的工作负载和严格的性能约束，通过建立一个适应性策略来实现最大化期望目标。 |

# 详细

[^1]: 一种用于智能自动缩放无服务器函数的深度循环强化学习方法

    A Deep Recurrent-Reinforcement Learning Method for Intelligent AutoScaling of Serverless Functions. (arXiv:2308.05937v1 [cs.DC])

    [http://arxiv.org/abs/2308.05937](http://arxiv.org/abs/2308.05937)

    该论文介绍了一种用于智能自动缩放无服务器函数的深度循环强化学习方法，针对波动的工作负载和严格的性能约束，通过建立一个适应性策略来实现最大化期望目标。

    

    函数即服务（FaaS）引入了一种轻量级的基于函数的云执行模型，在物联网边缘数据处理和异常检测等应用中具有相关性。虽然云服务提供商提供了几乎无限的函数弹性，但这些应用经常遇到波动的工作负载和更严格的性能约束。典型的云服务提供商策略是根据基于监控的阈值（如CPU或内存）来经验性地确定和调整所需的函数实例以适应需求和性能，即"自动缩放"。然而，阈值配置要么需要专家知识，要么需要历史数据或对环境的完整视图，使得自动缩放成为缺乏适应性解决方案的性能瓶颈。强化学习算法已被证明在分析复杂的云环境中是有益的，并产生适应性策略以最大化期望目标。

    Function-as-a-Service (FaaS) introduces a lightweight, function-based cloud execution model that finds its relevance in applications like IoT-edge data processing and anomaly detection. While CSP offer a near-infinite function elasticity, these applications often experience fluctuating workloads and stricter performance constraints. A typical CSP strategy is to empirically determine and adjust desired function instances, "autoscaling", based on monitoring-based thresholds such as CPU or memory, to cope with demand and performance. However, threshold configuration either requires expert knowledge, historical data or a complete view of environment, making autoscaling a performance bottleneck lacking an adaptable solution.RL algorithms are proven to be beneficial in analysing complex cloud environments and result in an adaptable policy that maximizes the expected objectives. Most realistic cloud environments usually involve operational interference and have limited visibility, making them
    

