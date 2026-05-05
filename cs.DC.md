# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Fair Multi-Agent Bandits.](http://arxiv.org/abs/2306.04498) | 本文针对多智能体之间公平多臂赌博机学习问题提出了一种算法，通过分布式拍卖算法学习样本最优匹配，使用一种新的利用阶段和一种基于顺序统计的遗憾分析实现，相较于先前的结果遗憾阶数从$O(\log T \log\log T)$到了$O\left(N^3 \log N \log T \right)$，能够更好地处理多个智能体之间的依赖关系。 |

# 详细

[^1]: 公平多智能体赌博机的最优算法研究

    Optimal Fair Multi-Agent Bandits. (arXiv:2306.04498v1 [cs.LG])

    [http://arxiv.org/abs/2306.04498](http://arxiv.org/abs/2306.04498)

    本文针对多智能体之间公平多臂赌博机学习问题提出了一种算法，通过分布式拍卖算法学习样本最优匹配，使用一种新的利用阶段和一种基于顺序统计的遗憾分析实现，相较于先前的结果遗憾阶数从$O(\log T \log\log T)$到了$O\left(N^3 \log N \log T \right)$，能够更好地处理多个智能体之间的依赖关系。

    

    本文研究了在多个不相互通信的智能体之间进行公平的多臂赌博机学习的问题，这些智能体只有在同时访问同一个臂时才提供碰撞信息。我们提出了一种算法，其遗憾为$O\left(N^3 \log N \log T \right)$（假设奖励有界，但未知上界）。这大大改进了之前结果，其遗憾阶数为$O(\log T \log\log T)$，并且对智能体数量具有指数依赖性。结果是通过使用分布式拍卖算法来学习样本最优匹配，一种新的利用阶段，其长度来自于观察到的样本，以及一种基于顺序统计的遗憾分析实现的。仿真结果显示了遗憾对$\log T$的依存关系。

    In this paper, we study the problem of fair multi-agent multi-arm bandit learning when agents do not communicate with each other, except collision information, provided to agents accessing the same arm simultaneously. We provide an algorithm with regret $O\left(N^3 \log N \log T \right)$ (assuming bounded rewards, with unknown bound). This significantly improves previous results which had regret of order $O(\log T \log\log T)$ and exponential dependence on the number of agents. The result is attained by using a distributed auction algorithm to learn the sample-optimal matching, a new type of exploitation phase whose length is derived from the observed samples, and a novel order-statistics-based regret analysis. Simulation results present the dependence of the regret on $\log T$.
    

