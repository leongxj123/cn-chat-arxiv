# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transferability of Graph Neural Networks using Graphon and Sampling Theories.](http://arxiv.org/abs/2307.13206) | 本文提出了一种新的方法来实现图神经网络的可迁移性，通过使用图谱和采样理论，我们证明了一个显式的两层图谱神经网络能够在保持准确性的同时以较少的网络权重数逼近带限信号，并且在收敛到图谱的序列中实现了在足够大的图之间的可迁移性。 |

# 详细

[^1]: 使用图论和采样理论实现图神经网络的可迁移性

    Transferability of Graph Neural Networks using Graphon and Sampling Theories. (arXiv:2307.13206v1 [cs.LG])

    [http://arxiv.org/abs/2307.13206](http://arxiv.org/abs/2307.13206)

    本文提出了一种新的方法来实现图神经网络的可迁移性，通过使用图谱和采样理论，我们证明了一个显式的两层图谱神经网络能够在保持准确性的同时以较少的网络权重数逼近带限信号，并且在收敛到图谱的序列中实现了在足够大的图之间的可迁移性。

    

    图神经网络（GNNs）已成为在各个领域处理基于图的信息的强大工具。GNN的一个理想特性是可迁移性，即训练好的网络可以在不重新训练的情况下交换来自不同图的信息并保持准确性。最近一种捕捉GNN可迁移性的方法是使用图谱，它是对大型稠密图的极限的对称可测函数。在这项工作中，我们通过提出一个显式的两层图谱神经网络（WNN）架构，对图谱应用于GNN做出了贡献。我们证明了它能够以指定误差容限在最少的网络权重数下逼近带限信号。然后，我们利用这一结果，在一个收敛到图谱的序列中，建立了一个明确的两层GNN在所有足够大的图之间的可迁移性。我们的工作解决了确定性加权图和简单随机图之间的可迁移性问题。

    Graph neural networks (GNNs) have become powerful tools for processing graph-based information in various domains. A desirable property of GNNs is transferability, where a trained network can swap in information from a different graph without retraining and retain its accuracy. A recent method of capturing transferability of GNNs is through the use of graphons, which are symmetric, measurable functions representing the limit of large dense graphs. In this work, we contribute to the application of graphons to GNNs by presenting an explicit two-layer graphon neural network (WNN) architecture. We prove its ability to approximate bandlimited signals within a specified error tolerance using a minimal number of network weights. We then leverage this result, to establish the transferability of an explicit two-layer GNN over all sufficiently large graphs in a sequence converging to a graphon. Our work addresses transferability between both deterministic weighted graphs and simple random graphs
    

