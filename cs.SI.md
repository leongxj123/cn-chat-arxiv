# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Random Geometric Graph Alignment with Graph Neural Networks](https://arxiv.org/abs/2402.07340) | 本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。 |
| [^2] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |
| [^3] | [Learning Hypergraphs From Signals With Dual Smoothness Prior.](http://arxiv.org/abs/2211.01717) | 本研究提出了一种基于双重平滑先验的超图结构学习框架，可从观察到的信号中学习超图结构以捕获实体间的内在高阶关系。 |

# 详细

[^1]: 用图神经网络对随机几何图进行对齐

    Random Geometric Graph Alignment with Graph Neural Networks

    [https://arxiv.org/abs/2402.07340](https://arxiv.org/abs/2402.07340)

    本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。

    

    我们研究了在顶点特征信息存在的情况下，图神经网络在图对齐问题中的性能。具体而言，给定两个独立扰动的单个随机几何图以及噪声稀疏特征的情况下，任务是恢复两个图的顶点之间的未知一对一映射关系。我们证明在特征向量的稀疏性和噪声水平满足一定条件的情况下，经过精心设计的单层图神经网络可以在很高的概率下通过图结构来恢复正确的顶点对齐。我们还证明了噪声水平的条件上界，仅存在对数因子差距。最后，我们将图神经网络的性能与直接在噪声顶点特征上求解分配问题进行了比较。我们证明了当噪声水平至少为常数时，这种直接匹配会导致恢复不完全，而图神经网络可以容忍n

    We characterize the performance of graph neural networks for graph alignment problems in the presence of vertex feature information. More specifically, given two graphs that are independent perturbations of a single random geometric graph with noisy sparse features, the task is to recover an unknown one-to-one mapping between the vertices of the two graphs. We show under certain conditions on the sparsity and noise level of the feature vectors, a carefully designed one-layer graph neural network can with high probability recover the correct alignment between the vertices with the help of the graph structure. We also prove that our conditions on the noise level are tight up to logarithmic factors. Finally we compare the performance of the graph neural network to directly solving an assignment problem on the noisy vertex features. We demonstrate that when the noise level is at least constant this direct matching fails to have perfect recovery while the graph neural network can tolerate n
    
[^2]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    
[^3]: 基于双重平滑先验学习信号的超图结构

    Learning Hypergraphs From Signals With Dual Smoothness Prior. (arXiv:2211.01717v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01717](http://arxiv.org/abs/2211.01717)

    本研究提出了一种基于双重平滑先验的超图结构学习框架，可从观察到的信号中学习超图结构以捕获实体间的内在高阶关系。

    

    超图结构学习是从观察到的信号中学习超图结构，以捕捉实体之间内在的高阶关系，当数据集中没有可用的超图拓扑结构时，这变得非常关键。本文提出了一种新的双重平滑先验的超图结构学习框架HGSL，通过把每个超边与具有节点信号平滑性和边连接性的子图对应起来，揭示了观察到的节点信号和超图结构之间的映射。实验结果表明了该方法的有效性。

    Hypergraph structure learning, which aims to learn the hypergraph structures from the observed signals to capture the intrinsic high-order relationships among the entities, becomes crucial when a hypergraph topology is not readily available in the datasets. There are two challenges that lie at the heart of this problem: 1) how to handle the huge search space of potential hyperedges, and 2) how to define meaningful criteria to measure the relationship between the signals observed on nodes and the hypergraph structure. In this paper, for the first challenge, we adopt the assumption that the ideal hypergraph structure can be derived from a learnable graph structure that captures the pairwise relations within signals. Further, we propose a hypergraph structure learning framework HGSL with a novel dual smoothness prior that reveals a mapping between the observed node signals and the hypergraph structure, whereby each hyperedge corresponds to a subgraph with both node signal smoothness and e
    

