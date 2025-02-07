# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partitioned Neural Network Training via Synthetic Intermediate Labels](https://arxiv.org/abs/2403.11204) | 该研究提出了一种通过将模型分区到不同GPU上，并生成合成中间标签来训练各个部分的方法，以缓解大规模神经网络训练中的内存和计算压力。 |

# 详细

[^1]: 通过合成中间标签进行分区神经网络训练

    Partitioned Neural Network Training via Synthetic Intermediate Labels

    [https://arxiv.org/abs/2403.11204](https://arxiv.org/abs/2403.11204)

    该研究提出了一种通过将模型分区到不同GPU上，并生成合成中间标签来训练各个部分的方法，以缓解大规模神经网络训练中的内存和计算压力。

    

    大规模神经网络架构的普及，特别是深度学习模型，对资源密集型训练提出了挑战。 GPU 内存约束已经成为训练这些庞大模型的一个明显瓶颈。现有策略，包括数据并行、模型并行、流水线并行和完全分片数据并行，提供了部分解决方案。 特别是模型并行允许将整个模型分布在多个 GPU 上，但随后的这些分区之间的数据通信减慢了训练速度。此外，为在每个 GPU 上存储辅助参数所需的大量内存开销增加了计算需求。 本研究主张不使用整个模型进行训练，而是将模型分区到 GPU 上，并生成合成中间标签来训练各个部分。 通过随机过程生成的这些标签减缓了训练中的内存和计算压力。

    arXiv:2403.11204v1 Announce Type: cross  Abstract: The proliferation of extensive neural network architectures, particularly deep learning models, presents a challenge in terms of resource-intensive training. GPU memory constraints have become a notable bottleneck in training such sizable models. Existing strategies, including data parallelism, model parallelism, pipeline parallelism, and fully sharded data parallelism, offer partial solutions. Model parallelism, in particular, enables the distribution of the entire model across multiple GPUs, yet the ensuing data communication between these partitions slows down training. Additionally, the substantial memory overhead required to store auxiliary parameters on each GPU compounds computational demands. Instead of using the entire model for training, this study advocates partitioning the model across GPUs and generating synthetic intermediate labels to train individual segments. These labels, produced through a random process, mitigate me
    

