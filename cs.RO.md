# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks](https://arxiv.org/abs/2402.17363) | CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题 |

# 详细

[^1]: CGGM：一种具有自适应稀疏性的条件图生成模型，用于物联网网络中节点异常检测

    CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks

    [https://arxiv.org/abs/2402.17363](https://arxiv.org/abs/2402.17363)

    CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题

    

    动态图被广泛用于检测物联网中节点的异常行为。生成模型通常用于解决动态图中节点类别不平衡的问题。然而，它面临的约束包括邻接关系的单调性，为节点构建多维特征的困难，以及缺乏端到端生成多类节点的方法。本文提出了一种名为CGGM的新颖图生成模型，专门设计用于生成少数类别中更多节点。通过自适应稀疏性生成邻接矩阵的机制增强了其结构的灵活性。特征生成模块名为多维特征生成器（MFG），可生成包括拓扑信息在内的节点特征。标签被转换为嵌入向量，用作条件。

    arXiv:2402.17363v1 Announce Type: cross  Abstract: Dynamic graphs are extensively employed for detecting anomalous behavior in nodes within the Internet of Things (IoT). Generative models are often used to address the issue of imbalanced node categories in dynamic graphs. Nevertheless, the constraints it faces include the monotonicity of adjacency relationships, the difficulty in constructing multi-dimensional features for nodes, and the lack of a method for end-to-end generation of multiple categories of nodes. This paper presents a novel graph generation model, called CGGM, designed specifically to generate a larger number of nodes belonging to the minority class. The mechanism for generating an adjacency matrix, through adaptive sparsity, enhances flexibility in its structure. The feature generation module, called multidimensional features generator (MFG) to generate node features along with topological information. Labels are transformed into embedding vectors, serving as condition
    

