# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Byzantine-Resilient Decentralized Multi-Armed Bandits.](http://arxiv.org/abs/2310.07320) | 这篇论文介绍了一种拜占庭容错的分散式多臂赌博机算法，通过信息混合和值的截断实现了对拜占庭代理的恢复和鲁棒性。 |

# 详细

[^1]: 拜占庭容错的分散式多臂赌博机算法

    Byzantine-Resilient Decentralized Multi-Armed Bandits. (arXiv:2310.07320v1 [cs.LG])

    [http://arxiv.org/abs/2310.07320](http://arxiv.org/abs/2310.07320)

    这篇论文介绍了一种拜占庭容错的分散式多臂赌博机算法，通过信息混合和值的截断实现了对拜占庭代理的恢复和鲁棒性。

    

    在分散式合作的多臂赌博机中，每个代理观察到不同的奖励流，试图与其他代理交换信息以选择一系列手臂以最小化遗憾。与独立运行上界置信度（UCB）等多臂赌博机方法相比，协作场景中的代理可以表现得更好。本文研究了如何在未知比例的代理可能是拜占庭（即，以奖励均值估计或置信度集的形式传递任意错误信息）时恢复此类突出行为。该框架可用于模拟计算机网络中的攻击者，向推荐系统中插入攻击性内容的策划者，或者金融市场的操纵者。我们的主要贡献是开发了一种完全分散的具有容错上界置信度（UCB）算法，该算法将代理间的信息混合步骤与不一致和极端值的截断相结合。这个截断步骤使我们能够建立

    In decentralized cooperative multi-armed bandits (MAB), each agent observes a distinct stream of rewards, and seeks to exchange information with others to select a sequence of arms so as to minimize its regret. Agents in the cooperative setting can outperform a single agent running a MAB method such as Upper-Confidence Bound (UCB) independently. In this work, we study how to recover such salient behavior when an unknown fraction of the agents can be Byzantine, that is, communicate arbitrarily wrong information in the form of reward mean-estimates or confidence sets. This framework can be used to model attackers in computer networks, instigators of offensive content into recommender systems, or manipulators of financial markets. Our key contribution is the development of a fully decentralized resilient upper confidence bound (UCB) algorithm that fuses an information mixing step among agents with a truncation of inconsistent and extreme values. This truncation step enables us to establis
    

