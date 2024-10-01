# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Multi-Agent Mapping for Planetary Exploration](https://arxiv.org/abs/2404.02289) | 联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。 |
| [^2] | [CEDAS: A Compressed Decentralized Stochastic Gradient Method with Improved Convergence](https://arxiv.org/abs/2301.05872) | CEDAS提出了一种压缩分布式随机梯度方法，在无偏压缩运算符下具有与集中式随机梯度下降相当的收敛速度，实现了最短的瞬态时间，对光滑强凸和非凸目标函数都适用。 |

# 详细

[^1]: 行星探测的联邦多智能体建图

    Federated Multi-Agent Mapping for Planetary Exploration

    [https://arxiv.org/abs/2404.02289](https://arxiv.org/abs/2404.02289)

    联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。

    

    在多智能体机器人探测中，管理和有效利用动态环境产生的大量异构数据构成了一个重要挑战。联邦学习（FL）是一种有前途的分布式映射方法，它解决了协作学习中去中心化数据的挑战。FL使多个智能体之间可以进行联合模型训练，而无需集中化或共享原始数据，克服了带宽和存储限制。我们的方法利用隐式神经映射，将地图表示为由神经网络学习的连续函数，以便实现紧凑和适应性的表示。我们进一步通过在地球数据集上进行元初始化来增强这一方法，预训练网络以快速学习新的地图结构。这种组合在诸如火星地形和冰川等不同领域展现了较强的泛化能力。我们对这一方法进行了严格评估，展示了其有效性。

    arXiv:2404.02289v1 Announce Type: cross  Abstract: In multi-agent robotic exploration, managing and effectively utilizing the vast, heterogeneous data generated from dynamic environments poses a significant challenge. Federated learning (FL) is a promising approach for distributed mapping, addressing the challenges of decentralized data in collaborative learning. FL enables joint model training across multiple agents without requiring the centralization or sharing of raw data, overcoming bandwidth and storage constraints. Our approach leverages implicit neural mapping, representing maps as continuous functions learned by neural networks, for compact and adaptable representations. We further enhance this approach with meta-initialization on Earth datasets, pre-training the network to quickly learn new map structures. This combination demonstrates strong generalization to diverse domains like Martian terrain and glaciers. We rigorously evaluate this approach, demonstrating its effectiven
    
[^2]: CEDAS：一种具有改进收敛性的压缩分布式随机梯度法

    CEDAS: A Compressed Decentralized Stochastic Gradient Method with Improved Convergence

    [https://arxiv.org/abs/2301.05872](https://arxiv.org/abs/2301.05872)

    CEDAS提出了一种压缩分布式随机梯度方法，在无偏压缩运算符下具有与集中式随机梯度下降相当的收敛速度，实现了最短的瞬态时间，对光滑强凸和非凸目标函数都适用。

    

    在本文中，我们考虑在通信受限环境下解决多代理网络上的分布式优化问题。我们研究了一种称为“具有自适应步长的压缩精确扩散（CEDAS）”的压缩分布式随机梯度方法，并证明该方法在无偏压缩运算符下渐近地实现了与集中式随机梯度下降（SGD）相当的收敛速度，适用于光滑强凸目标函数和光滑非凸目标函数。特别地，据我们所知，CEDAS迄今为止以其最短的瞬态时间（关于图的特性）实现了与集中式SGD相同的收敛速度，其在光滑强凸目标函数下表现为$\mathcal{O}(n{C^3}/(1-\lambda_2)^{2})$，在光滑非凸目标函数下表现为$\mathcal{O}(n^3{C^6}/(1-\lambda_2)^4)$，其中$(1-\lambda_2)$表示谱...

    arXiv:2301.05872v2 Announce Type: replace-cross  Abstract: In this paper, we consider solving the distributed optimization problem over a multi-agent network under the communication restricted setting. We study a compressed decentralized stochastic gradient method, termed ``compressed exact diffusion with adaptive stepsizes (CEDAS)", and show the method asymptotically achieves comparable convergence rate as centralized { stochastic gradient descent (SGD)} for both smooth strongly convex objective functions and smooth nonconvex objective functions under unbiased compression operators. In particular, to our knowledge, CEDAS enjoys so far the shortest transient time (with respect to the graph specifics) for achieving the convergence rate of centralized SGD, which behaves as $\mathcal{O}(n{C^3}/(1-\lambda_2)^{2})$ under smooth strongly convex objective functions, and $\mathcal{O}(n^3{C^6}/(1-\lambda_2)^4)$ under smooth nonconvex objective functions, where $(1-\lambda_2)$ denotes the spectr
    

