# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Faster Convergence with Less Communication: Broadcast-Based Subgraph Sampling for Decentralized Learning over Wireless Networks.](http://arxiv.org/abs/2401.13779) | 本文提出了一种名为BASS的基于广播的子图采样方法，用于加速去中心化学习算法的收敛速度，并减少通信成本。 |

# 详细

[^1]: 更快的收敛速度和更少的通信成本：用于无线网络的基于广播的子图采样的去中心化学习

    Faster Convergence with Less Communication: Broadcast-Based Subgraph Sampling for Decentralized Learning over Wireless Networks. (arXiv:2401.13779v1 [cs.IT])

    [http://arxiv.org/abs/2401.13779](http://arxiv.org/abs/2401.13779)

    本文提出了一种名为BASS的基于广播的子图采样方法，用于加速去中心化学习算法的收敛速度，并减少通信成本。

    

    基于共识的去中心化随机梯度下降(D-SGD)是一种广泛采用的算法，用于网络代理之间的去中心化机器学习模型训练。D-SGD的一个关键部分是基于共识的模型平均，它严重依赖于节点之间的信息交换和融合。特别地，对于在无线网络上的共识平均，通信协调是必要的，以确定节点何时以及如何访问信道，并将信息传输（或接收）给（或从）邻居节点。在这项工作中，我们提出了一种名为BASS的基于广播的子图采样方法，旨在加快D-SGD的收敛速度，并考虑每轮迭代的实际通信成本。BASS创建一组混合矩阵候选项，表示基础拓扑的稀疏子图。在每个共识迭代中，将采样一个混合矩阵，从而产生一个特定的调度决策，激活多个无碰撞的节点子集。

    Consensus-based decentralized stochastic gradient descent (D-SGD) is a widely adopted algorithm for decentralized training of machine learning models across networked agents. A crucial part of D-SGD is the consensus-based model averaging, which heavily relies on information exchange and fusion among the nodes. Specifically, for consensus averaging over wireless networks, communication coordination is necessary to determine when and how a node can access the channel and transmit (or receive) information to (or from) its neighbors. In this work, we propose $\texttt{BASS}$, a broadcast-based subgraph sampling method designed to accelerate the convergence of D-SGD while considering the actual communication cost per iteration. $\texttt{BASS}$ creates a set of mixing matrix candidates that represent sparser subgraphs of the base topology. In each consensus iteration, one mixing matrix is sampled, leading to a specific scheduling decision that activates multiple collision-free subsets of node
    

