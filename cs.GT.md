# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators](https://arxiv.org/abs/2403.15524) | 引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。 |
| [^2] | [Offline Fictitious Self-Play for Competitive Games](https://arxiv.org/abs/2403.00841) | 本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。 |

# 详细

[^1]: PPA-Game：表征和学习在线内容创作者之间的竞争动态

    PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators

    [https://arxiv.org/abs/2403.15524](https://arxiv.org/abs/2403.15524)

    引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。

    

    我们引入了比例性收益分配游戏（PPA-Game）来模拟代理者如何竞争可分配资源和消费者的注意力，类似于YouTube和TikTok等平台上的内容创作者。根据异质权重为代理者分配收益，反映了创作者之间内容质量的多样性。我们的分析表明，纯纳什均衡（PNE）并不在每种情况下都有保证，但在我们的模拟中，通常会观察到，其缺乏情况是罕见的。除了分析静态收益外，我们进一步讨论了代理者关于资源收益的在线学习，将多玩家多臂老虎机框架整合在一起。我们提出了一种在线算法，在$T$轮中促进每个代理者累积收益的最大化。从理论上讲，我们建立了任何代理者的遗憾在任何$\eta > 0$下都受到$O(\log^{1 + \eta} T)$的限制。经验结果进一步验证了我们的算法的有效性。

    arXiv:2403.15524v1 Announce Type: cross  Abstract: We introduce the Proportional Payoff Allocation Game (PPA-Game) to model how agents, akin to content creators on platforms like YouTube and TikTok, compete for divisible resources and consumers' attention. Payoffs are allocated to agents based on heterogeneous weights, reflecting the diversity in content quality among creators. Our analysis reveals that although a pure Nash equilibrium (PNE) is not guaranteed in every scenario, it is commonly observed, with its absence being rare in our simulations. Beyond analyzing static payoffs, we further discuss the agents' online learning about resource payoffs by integrating a multi-player multi-armed bandit framework. We propose an online algorithm facilitating each agent's maximization of cumulative payoffs over $T$ rounds. Theoretically, we establish that the regret of any agent is bounded by $O(\log^{1 + \eta} T)$ for any $\eta > 0$. Empirical results further validate the effectiveness of ou
    
[^2]: 竞争游戏的离线虚构自我对弈

    Offline Fictitious Self-Play for Competitive Games

    [https://arxiv.org/abs/2403.00841](https://arxiv.org/abs/2403.00841)

    本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。

    

    离线强化学习（RL）因其在以前收集的数据集中改进策略而不需要在线交互的能力而受到重视。尽管在单一智能体设置中取得成功，但离线多智能体RL仍然是一个挑战，特别是在竞争游戏中。为了解决这些问题，本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法。我们首先通过调整固定数据集的权重，使用重要性抽样模拟与各种对手的互动。

    arXiv:2403.00841v1 Announce Type: cross  Abstract: Offline Reinforcement Learning (RL) has received significant interest due to its ability to improve policies in previously collected datasets without online interactions. Despite its success in the single-agent setting, offline multi-agent RL remains a challenge, especially in competitive games. Firstly, unaware of the game structure, it is impossible to interact with the opponents and conduct a major learning paradigm, self-play, for competitive games. Secondly, real-world datasets cannot cover all the state and action space in the game, resulting in barriers to identifying Nash equilibrium (NE). To address these issues, this paper introduces Off-FSP, the first practical model-free offline RL algorithm for competitive games. We start by simulating interactions with various opponents by adjusting the weights of the fixed dataset with importance sampling. This technique allows us to learn best responses to different opponents and employ
    

