# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators](https://arxiv.org/abs/2403.15524) | 引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。 |

# 详细

[^1]: PPA-Game：表征和学习在线内容创作者之间的竞争动态

    PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators

    [https://arxiv.org/abs/2403.15524](https://arxiv.org/abs/2403.15524)

    引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。

    

    我们引入了比例性收益分配游戏（PPA-Game）来模拟代理者如何竞争可分配资源和消费者的注意力，类似于YouTube和TikTok等平台上的内容创作者。根据异质权重为代理者分配收益，反映了创作者之间内容质量的多样性。我们的分析表明，纯纳什均衡（PNE）并不在每种情况下都有保证，但在我们的模拟中，通常会观察到，其缺乏情况是罕见的。除了分析静态收益外，我们进一步讨论了代理者关于资源收益的在线学习，将多玩家多臂老虎机框架整合在一起。我们提出了一种在线算法，在$T$轮中促进每个代理者累积收益的最大化。从理论上讲，我们建立了任何代理者的遗憾在任何$\eta > 0$下都受到$O(\log^{1 + \eta} T)$的限制。经验结果进一步验证了我们的算法的有效性。

    arXiv:2403.15524v1 Announce Type: cross  Abstract: We introduce the Proportional Payoff Allocation Game (PPA-Game) to model how agents, akin to content creators on platforms like YouTube and TikTok, compete for divisible resources and consumers' attention. Payoffs are allocated to agents based on heterogeneous weights, reflecting the diversity in content quality among creators. Our analysis reveals that although a pure Nash equilibrium (PNE) is not guaranteed in every scenario, it is commonly observed, with its absence being rare in our simulations. Beyond analyzing static payoffs, we further discuss the agents' online learning about resource payoffs by integrating a multi-player multi-armed bandit framework. We propose an online algorithm facilitating each agent's maximization of cumulative payoffs over $T$ rounds. Theoretically, we establish that the regret of any agent is bounded by $O(\log^{1 + \eta} T)$ for any $\eta > 0$. Empirical results further validate the effectiveness of ou
    

