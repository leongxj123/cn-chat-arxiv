# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Split Learning over Energy-Constrained Wireless Edge Networks](https://arxiv.org/abs/2403.05158) | 设计了一种在无线边缘网络中为设备动态选择分裂点并为服务器分配计算资源的自适应分裂学习方案，以最小化平均训练延迟为目标，并提出了一种名为OPEN的在线算法解决此问题。 |

# 详细

[^1]: 能量受限的无线边缘网络中的自适应分裂学习

    Adaptive Split Learning over Energy-Constrained Wireless Edge Networks

    [https://arxiv.org/abs/2403.05158](https://arxiv.org/abs/2403.05158)

    设计了一种在无线边缘网络中为设备动态选择分裂点并为服务器分配计算资源的自适应分裂学习方案，以最小化平均训练延迟为目标，并提出了一种名为OPEN的在线算法解决此问题。

    

    分裂学习（SL）是一种有希望的用于训练人工智能（AI）模型的方法，其中设备与服务器合作以分布式方式训练AI模型，基于相同的固定分裂点。然而，由于设备的异构性和信道条件的变化，这种方式在训练延迟和能量消耗方面并不是最优的。在本文中，我们设计了一种自适应分裂学习（ASL）方案，可以在无线边缘网络中为设备动态选择分裂点，并为服务器分配计算资源。我们制定了一个优化问题，旨在在满足长期能量消耗约束的情况下最小化平均训练延迟。解决这个问题的困难在于缺乏未来信息和混合整数规划（MIP）。为了解决这个问题，我们提出了一种利用Lyapunov理论的在线算法，名为OPEN，它将其分解为一个具有当前的新MIP问题。

    arXiv:2403.05158v1 Announce Type: cross  Abstract: Split learning (SL) is a promising approach for training artificial intelligence (AI) models, in which devices collaborate with a server to train an AI model in a distributed manner, based on a same fixed split point. However, due to the device heterogeneity and variation of channel conditions, this way is not optimal in training delay and energy consumption. In this paper, we design an adaptive split learning (ASL) scheme which can dynamically select split points for devices and allocate computing resource for the server in wireless edge networks. We formulate an optimization problem to minimize the average training latency subject to long-term energy consumption constraint. The difficulties in solving this problem are the lack of future information and mixed integer programming (MIP). To solve it, we propose an online algorithm leveraging the Lyapunov theory, named OPEN, which decomposes it into a new MIP problem only with the curren
    

