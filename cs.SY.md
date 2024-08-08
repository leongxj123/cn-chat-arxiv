# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Point-based Value Iteration for Neuro-Symbolic POMDPs.](http://arxiv.org/abs/2306.17639) | 本文提出了一种基于点的神经符号POMDP值迭代算法，结合了传统符号技术和神经网络，解决了连续状态置信度函数的问题，实现了优化折扣累积回报的连续状态决策问题。 |

# 详细

[^1]: 基于点的神经符号POMDP值迭代

    Point-based Value Iteration for Neuro-Symbolic POMDPs. (arXiv:2306.17639v1 [eess.SY])

    [http://arxiv.org/abs/2306.17639](http://arxiv.org/abs/2306.17639)

    本文提出了一种基于点的神经符号POMDP值迭代算法，结合了传统符号技术和神经网络，解决了连续状态置信度函数的问题，实现了优化折扣累积回报的连续状态决策问题。

    

    神经符号人工智能是结合传统符号技术和神经网络的新兴领域。本文考虑其在不确定性下顺序决策中的应用。我们引入了神经符号部分可观察马尔可夫决策过程（NS-POMDPs），该模型描述了一个使用神经网络感知连续状态环境并进行符号决策的代理，并研究了优化折扣累积回报的问题。针对连续状态置信度函数，我们提出了一种新的分段线性和凸表示（P-PWLC），通过覆盖连续状态空间的多面体和值向量实现，并将Bellman backups扩展到该表示。我们证明了值函数的凸性和连续性，并提出了两种值迭代算法，通过利用连续状态模型和神经感知机制的底层结构来保证有限表示能力。

    Neuro-symbolic artificial intelligence is an emerging area that combines traditional symbolic techniques with neural networks. In this paper, we consider its application to sequential decision making under uncertainty. We introduce neuro-symbolic partially observable Markov decision processes (NS-POMDPs), which model an agent that perceives a continuous-state environment using a neural network and makes decisions symbolically, and study the problem of optimising discounted cumulative rewards. This requires functions over continuous-state beliefs, for which we propose a novel piecewise linear and convex representation (P-PWLC) in terms of polyhedra covering the continuous-state space and value vectors, and extend Bellman backups to this representation. We prove the convexity and continuity of value functions and present two value iteration algorithms that ensure finite representability by exploiting the underlying structure of the continuous-state model and the neural perception mechani
    

