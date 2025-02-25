# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Communication-Efficient Federated Optimization over Semi-Decentralized Networks.](http://arxiv.org/abs/2311.18787) | 本文提出了一种基于半分散网络的通信高效算法PISCO, 通过概率性的代理间和代理与服务器之间的通信，实现了通信效率与收敛速度的折衷。 |

# 详细

[^1]: 通信高效的半分散网络联邦优化

    Communication-Efficient Federated Optimization over Semi-Decentralized Networks. (arXiv:2311.18787v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.18787](http://arxiv.org/abs/2311.18787)

    本文提出了一种基于半分散网络的通信高效算法PISCO, 通过概率性的代理间和代理与服务器之间的通信，实现了通信效率与收敛速度的折衷。

    

    在大规模的联邦和分散式学习中，通信效率是最具挑战性的瓶颈之一。本文提出了一种半分散通信协议下的通信高效算法PISCO，通过概率性的代理间和代理与服务器之间的通信，实现了通信效率与收敛速度的折衷。PISCO算法通过梯度追踪和多个本地更新保证了对数据异质性的鲁棒性。我们证明了PISCO算法在非凸问题上的收敛速度，并展示了在数量方面，PISCO算法具有线性加速的优势。

    In large-scale federated and decentralized learning, communication efficiency is one of the most challenging bottlenecks. While gossip communication -- where agents can exchange information with their connected neighbors -- is more cost-effective than communicating with the remote server, it often requires a greater number of communication rounds, especially for large and sparse networks. To tackle the trade-off, we examine the communication efficiency under a semi-decentralized communication protocol, in which agents can perform both agent-to-agent and agent-to-server communication in a probabilistic manner. We design a tailored communication-efficient algorithm over semi-decentralized networks, referred to as PISCO, which inherits the robustness to data heterogeneity thanks to gradient tracking and allows multiple local updates for saving communication. We establish the convergence rate of PISCO for nonconvex problems and show that PISCO enjoys a linear speedup in terms of the number
    

