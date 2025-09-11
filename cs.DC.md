# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models](https://arxiv.org/abs/2403.09904) | FedComLoc利用Scaffnew算法的基础，引入了压缩和本地训练，显著降低了分布式训练中的通信开销。 |

# 详细

[^1]: FedComLoc: 稀疏和量化模型的通信高效分布式训练

    FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models

    [https://arxiv.org/abs/2403.09904](https://arxiv.org/abs/2403.09904)

    FedComLoc利用Scaffnew算法的基础，引入了压缩和本地训练，显著降低了分布式训练中的通信开销。

    

    联邦学习（FL）由于其允许异构客户端在本地处理其私有数据并与中央服务器互动，同时尊重隐私的独特特点而受到越来越多的关注。我们的工作受到了创新的Scaffnew算法的启发，该算法在FL中大大推动了通信复杂性的降低。我们引入了FedComLoc（联邦压缩和本地训练），将实用且有效的压缩集成到Scaffnew中，以进一步增强通信效率。广泛的实验证明，使用流行的TopK压缩器和量化，它在大幅减少异构中的通信开销方面具有卓越的性能。

    arXiv:2403.09904v1 Announce Type: cross  Abstract: Federated Learning (FL) has garnered increasing attention due to its unique characteristic of allowing heterogeneous clients to process their private data locally and interact with a central server, while being respectful of privacy. A critical bottleneck in FL is the communication cost. A pivotal strategy to mitigate this burden is \emph{Local Training}, which involves running multiple local stochastic gradient descent iterations between communication phases. Our work is inspired by the innovative \emph{Scaffnew} algorithm, which has considerably advanced the reduction of communication complexity in FL. We introduce FedComLoc (Federated Compressed and Local Training), integrating practical and effective compression into \emph{Scaffnew} to further enhance communication efficiency. Extensive experiments, using the popular TopK compressor and quantization, demonstrate its prowess in substantially reducing communication overheads in heter
    

