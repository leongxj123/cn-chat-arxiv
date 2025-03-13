# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Random Geometric Graph Alignment with Graph Neural Networks](https://arxiv.org/abs/2402.07340) | 本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。 |

# 详细

[^1]: 用图神经网络对随机几何图进行对齐

    Random Geometric Graph Alignment with Graph Neural Networks

    [https://arxiv.org/abs/2402.07340](https://arxiv.org/abs/2402.07340)

    本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。

    

    我们研究了在顶点特征信息存在的情况下，图神经网络在图对齐问题中的性能。具体而言，给定两个独立扰动的单个随机几何图以及噪声稀疏特征的情况下，任务是恢复两个图的顶点之间的未知一对一映射关系。我们证明在特征向量的稀疏性和噪声水平满足一定条件的情况下，经过精心设计的单层图神经网络可以在很高的概率下通过图结构来恢复正确的顶点对齐。我们还证明了噪声水平的条件上界，仅存在对数因子差距。最后，我们将图神经网络的性能与直接在噪声顶点特征上求解分配问题进行了比较。我们证明了当噪声水平至少为常数时，这种直接匹配会导致恢复不完全，而图神经网络可以容忍n

    We characterize the performance of graph neural networks for graph alignment problems in the presence of vertex feature information. More specifically, given two graphs that are independent perturbations of a single random geometric graph with noisy sparse features, the task is to recover an unknown one-to-one mapping between the vertices of the two graphs. We show under certain conditions on the sparsity and noise level of the feature vectors, a carefully designed one-layer graph neural network can with high probability recover the correct alignment between the vertices with the help of the graph structure. We also prove that our conditions on the noise level are tight up to logarithmic factors. Finally we compare the performance of the graph neural network to directly solving an assignment problem on the noisy vertex features. We demonstrate that when the noise level is at least constant this direct matching fails to have perfect recovery while the graph neural network can tolerate n
    

