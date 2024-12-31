# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralized Multi-Armed Bandits Can Outperform Centralized Upper Confidence Bound Algorithms.](http://arxiv.org/abs/2111.10933) | 本文研究了一个多智能体网络中的分布式多臂赌博机问题，提出了两种完全分布式的算法，基于经典的UCB算法和KL-UCB算法，实验证明这些算法能达到更好的对数渐近后悔，智能体之间的邻居关系越多，后悔值越好。 |

# 详细

[^1]: 分布式多臂赌博机可以超越集中式上置信界限算法

    Decentralized Multi-Armed Bandits Can Outperform Centralized Upper Confidence Bound Algorithms. (arXiv:2111.10933v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.10933](http://arxiv.org/abs/2111.10933)

    本文研究了一个多智能体网络中的分布式多臂赌博机问题，提出了两种完全分布式的算法，基于经典的UCB算法和KL-UCB算法，实验证明这些算法能达到更好的对数渐近后悔，智能体之间的邻居关系越多，后悔值越好。

    

    本文研究了一个多智能体网络中的分布式多臂赌博机问题。假设N个智能体同时解决了这个问题，他们面对着一组共同的M个臂并共享相同的臂奖励分布。每个智能体只能从邻居处接收信息，智能体之间的邻居关系由一个无向图描述。本文提出了两种完全分布式的多臂赌博机算法，分别基于经典的上置信界限（UCB）算法和最先进的KL-UCB算法。所提出的分布式算法使网络中的每个智能体能够实现比其单一智能体相应算法更好的对数渐近后悔，前提是智能体至少有一个邻居，而且智能体有越多的邻居，后悔值会越好，这意味着整体的和大于其组成部分。

    This paper studies a decentralized multi-armed bandit problem in a multi-agent network. The problem is simultaneously solved by N agents assuming they face a common set of M arms and share the same arms' reward distributions. Each agent can receive information only from its neighbors, where the neighbor relationships among the agents are described by an undirected graph. Two fully decentralized multi-armed bandit algorithms are proposed, respectively based on the classic upper confidence bound (UCB) algorithm and the state-of-the-art KL-UCB algorithm. The proposed decentralized algorithms permit each agent in the network to achieve a better logarithmic asymptotic regret than their single-agent counterparts, provided that the agent has at least one neighbor, and the more neighbors an agent has, the better regret it will have, meaning that the sum is more than its component parts.
    

