# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs.](http://arxiv.org/abs/2401.13054) | 本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。 |

# 详细

[^1]: 无计算困难的快速计算超图节点距离的方法

    Frustrated Random Walks: A Fast Method to Compute Node Distances on Hypergraphs. (arXiv:2401.13054v1 [cs.SI])

    [http://arxiv.org/abs/2401.13054](http://arxiv.org/abs/2401.13054)

    本文提出了一种基于随机游走的方法，用于快速计算超图节点之间的距离并进行标签传播。该方法解决了超图中节点距离计算的问题，进一步拓展了超图的应用领域。

    

    超图是图的推广，当考虑实体间的属性共享时会自然产生。尽管可以通过将超边扩展为完全连接的子图来将超图转换为图，但逆向操作在计算上非常复杂且属于NP-complete问题。因此，我们假设超图包含比图更多的信息。此外，直接操作超图比将其扩展为图更为方便。超图中的一个开放问题是如何精确高效地计算节点之间的距离。通过估计节点距离，我们能够找到节点的最近邻居，并使用K最近邻（KNN）方法在超图上执行标签传播。在本文中，我们提出了一种基于随机游走的新方法，实现了在超图上进行标签传播。我们将节点距离估计为随机游走的预期到达时间。我们注意到简单随机游走（SRW）无法准确描述节点之间的距离，因此我们引入了"frustrated"的概念。

    A hypergraph is a generalization of a graph that arises naturally when attribute-sharing among entities is considered. Although a hypergraph can be converted into a graph by expanding its hyperedges into fully connected subgraphs, going the reverse way is computationally complex and NP-complete. We therefore hypothesize that a hypergraph contains more information than a graph. In addition, it is more convenient to manipulate a hypergraph directly, rather than expand it into a graph. An open problem in hypergraphs is how to accurately and efficiently calculate their node distances. Estimating node distances enables us to find a node's nearest neighbors, and perform label propagation on hypergraphs using a K-nearest neighbors (KNN) approach. In this paper, we propose a novel approach based on random walks to achieve label propagation on hypergraphs. We estimate node distances as the expected hitting times of random walks. We note that simple random walks (SRW) cannot accurately describe 
    

