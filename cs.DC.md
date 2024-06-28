# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism.](http://arxiv.org/abs/2303.13775) | 本文提出了一种新的并行小批量训练方法，即分裂并行，应用在图神经网络训练上，能有效缓解数据并行方法的性能瓶颈，同时在大规模图上的性能表现优越。 |

# 详细

[^1]: GSplit: 通过分裂并行实现大规模图神经网络训练的扩展

    GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism. (arXiv:2303.13775v1 [cs.DC])

    [http://arxiv.org/abs/2303.13775](http://arxiv.org/abs/2303.13775)

    本文提出了一种新的并行小批量训练方法，即分裂并行，应用在图神经网络训练上，能有效缓解数据并行方法的性能瓶颈，同时在大规模图上的性能表现优越。

    

    在许多行业、科学和工程领域（如推荐系统、社交图分析、知识库、材料科学和生物学）中，拥有数十亿个边的大规模图形是普遍存在的。图神经网络（GNN）作为一种新兴的机器学习模型，由于在各种图分析任务中具有卓越的性能，因此越来越多地被采用来学习这些图形。在大型图形上训练通常采用小批量训练，并且数据并行是将小批量训练扩展到多个 GPU 的标准方法。本文认为，GNN 训练系统的几个基本性能瓶颈与数据并行方法的固有限制有关。我们提出了一种新的并行小批量训练范式- 分裂并行，并将其实现在一个名为gsplit的新系统中。实验表明，gsplit 的性能优于DGL、Quiver和PaGraph等现有的系统。

    Large-scale graphs with billions of edges are ubiquitous in many industries, science, and engineering fields such as recommendation systems, social graph analysis, knowledge base, material science, and biology. Graph neural networks (GNN), an emerging class of machine learning models, are increasingly adopted to learn on these graphs due to their superior performance in various graph analytics tasks. Mini-batch training is commonly adopted to train on large graphs, and data parallelism is the standard approach to scale mini-batch training to multiple GPUs. In this paper, we argue that several fundamental performance bottlenecks of GNN training systems have to do with inherent limitations of the data parallel approach. We then propose split parallelism, a novel parallel mini-batch training paradigm. We implement split parallelism in a novel system called gsplit and show that it outperforms state-of-the-art systems such as DGL, Quiver, and PaGraph.
    

