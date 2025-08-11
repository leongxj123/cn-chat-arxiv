# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |

# 详细

[^1]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    

