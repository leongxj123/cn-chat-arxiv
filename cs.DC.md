# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faster Convergence with Less Communication: Broadcast-Based Subgraph Sampling for Decentralized Learning over Wireless Networks.](http://arxiv.org/abs/2401.13779) | 本文提出了一种名为BASS的基于广播的子图采样方法，用于加速去中心化学习算法的收敛速度，并减少通信成本。 |
| [^2] | [Communication-Efficient Federated Optimization over Semi-Decentralized Networks.](http://arxiv.org/abs/2311.18787) | 本文提出了一种基于半分散网络的通信高效算法PISCO, 通过概率性的代理间和代理与服务器之间的通信，实现了通信效率与收敛速度的折衷。 |

# 详细

[^1]: 更快的收敛速度和更少的通信成本：用于无线网络的基于广播的子图采样的去中心化学习

    Faster Convergence with Less Communication: Broadcast-Based Subgraph Sampling for Decentralized Learning over Wireless Networks. (arXiv:2401.13779v1 [cs.IT])

    [http://arxiv.org/abs/2401.13779](http://arxiv.org/abs/2401.13779)

    本文提出了一种名为BASS的基于广播的子图采样方法，用于加速去中心化学习算法的收敛速度，并减少通信成本。

    

    基于共识的去中心化随机梯度下降(D-SGD)是一种广泛采用的算法，用于网络代理之间的去中心化机器学习模型训练。D-SGD的一个关键部分是基于共识的模型平均，它严重依赖于节点之间的信息交换和融合。特别地，对于在无线网络上的共识平均，通信协调是必要的，以确定节点何时以及如何访问信道，并将信息传输（或接收）给（或从）邻居节点。在这项工作中，我们提出了一种名为BASS的基于广播的子图采样方法，旨在加快D-SGD的收敛速度，并考虑每轮迭代的实际通信成本。BASS创建一组混合矩阵候选项，表示基础拓扑的稀疏子图。在每个共识迭代中，将采样一个混合矩阵，从而产生一个特定的调度决策，激活多个无碰撞的节点子集。

    Consensus-based decentralized stochastic gradient descent (D-SGD) is a widely adopted algorithm for decentralized training of machine learning models across networked agents. A crucial part of D-SGD is the consensus-based model averaging, which heavily relies on information exchange and fusion among the nodes. Specifically, for consensus averaging over wireless networks, communication coordination is necessary to determine when and how a node can access the channel and transmit (or receive) information to (or from) its neighbors. In this work, we propose $\texttt{BASS}$, a broadcast-based subgraph sampling method designed to accelerate the convergence of D-SGD while considering the actual communication cost per iteration. $\texttt{BASS}$ creates a set of mixing matrix candidates that represent sparser subgraphs of the base topology. In each consensus iteration, one mixing matrix is sampled, leading to a specific scheduling decision that activates multiple collision-free subsets of node
    
[^2]: 通信高效的半分散网络联邦优化

    Communication-Efficient Federated Optimization over Semi-Decentralized Networks. (arXiv:2311.18787v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.18787](http://arxiv.org/abs/2311.18787)

    本文提出了一种基于半分散网络的通信高效算法PISCO, 通过概率性的代理间和代理与服务器之间的通信，实现了通信效率与收敛速度的折衷。

    

    在大规模的联邦和分散式学习中，通信效率是最具挑战性的瓶颈之一。本文提出了一种半分散通信协议下的通信高效算法PISCO，通过概率性的代理间和代理与服务器之间的通信，实现了通信效率与收敛速度的折衷。PISCO算法通过梯度追踪和多个本地更新保证了对数据异质性的鲁棒性。我们证明了PISCO算法在非凸问题上的收敛速度，并展示了在数量方面，PISCO算法具有线性加速的优势。

    In large-scale federated and decentralized learning, communication efficiency is one of the most challenging bottlenecks. While gossip communication -- where agents can exchange information with their connected neighbors -- is more cost-effective than communicating with the remote server, it often requires a greater number of communication rounds, especially for large and sparse networks. To tackle the trade-off, we examine the communication efficiency under a semi-decentralized communication protocol, in which agents can perform both agent-to-agent and agent-to-server communication in a probabilistic manner. We design a tailored communication-efficient algorithm over semi-decentralized networks, referred to as PISCO, which inherits the robustness to data heterogeneity thanks to gradient tracking and allows multiple local updates for saving communication. We establish the convergence rate of PISCO for nonconvex problems and show that PISCO enjoys a linear speedup in terms of the number
    

