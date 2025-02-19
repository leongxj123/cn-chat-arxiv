# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approximate Multiagent Reinforcement Learning for On-Demand Urban Mobility Problem on a Large Map (extended version).](http://arxiv.org/abs/2311.01534) | 本文研究了大型城市环境下的自主多智能体出租车路径问题，提出了一个近似滚动为基础的两阶段算法来减少计算量。 |

# 详细

[^1]: 大型地图上的按需城市出行问题的近似多智能体强化学习（扩展版）

    Approximate Multiagent Reinforcement Learning for On-Demand Urban Mobility Problem on a Large Map (extended version). (arXiv:2311.01534v1 [cs.MA])

    [http://arxiv.org/abs/2311.01534](http://arxiv.org/abs/2311.01534)

    本文研究了大型城市环境下的自主多智能体出租车路径问题，提出了一个近似滚动为基础的两阶段算法来减少计算量。

    

    本文关注大型城市环境下的自主多智能体出租车路径问题，未来乘车请求的位置和数量事先未知，但遵循估计的经验分布。最近的理论表明，如果基础策略是稳定的，那么基于滚动的算法与这样的基础策略产生接近最优的稳定策略。尽管基于滚动的方法非常适合学习具有对未来需求考虑的合作多智能体策略，但将这些方法应用于大型城市环境可能计算上很昂贵。大型环境往往有大量请求，因此需要大型的出租车队保证稳定性。本文旨在解决多智能体（逐一）滚动的计算瓶颈问题，其中计算复杂性随代理数量线性增长。我们提出了一种近似逐一滚动为基础的两阶段算法，减少计算量

    In this paper, we focus on the autonomous multiagent taxi routing problem for a large urban environment where the location and number of future ride requests are unknown a-priori, but follow an estimated empirical distribution. Recent theory has shown that if a base policy is stable then a rollout-based algorithm with such a base policy produces a near-optimal stable policy. Although, rollout-based approaches are well-suited for learning cooperative multiagent policies with considerations for future demand, applying such methods to a large urban environment can be computationally expensive. Large environments tend to have a large volume of requests, and hence require a large fleet of taxis to guarantee stability. In this paper, we aim to address the computational bottleneck of multiagent (one-at-a-time) rollout, where the computational complexity grows linearly in the number of agents. We propose an approximate one-at-a-time rollout-based two-phase algorithm that reduces the computatio
    

