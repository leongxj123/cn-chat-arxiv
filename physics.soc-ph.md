# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Het-node2vec: second order random walk sampling for heterogeneous multigraphs embedding.](http://arxiv.org/abs/2101.01425) | Het-node2vec是一个算法框架，通过在异构多图上进行二阶随机游走采样，能够捕获图的结构特征和不同类型节点边的语义，有效地提高对异构图的无监督和有监督学习性能。 |

# 详细

[^1]: Het-node2vec：异构多图嵌入的二阶随机游走采样方法

    Het-node2vec: second order random walk sampling for heterogeneous multigraphs embedding. (arXiv:2101.01425v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2101.01425](http://arxiv.org/abs/2101.01425)

    Het-node2vec是一个算法框架，通过在异构多图上进行二阶随机游走采样，能够捕获图的结构特征和不同类型节点边的语义，有效地提高对异构图的无监督和有监督学习性能。

    

    在多个真实世界应用中，为异构图开发图表示学习方法是基础性的，因为在多个上下文中，图由不同类型的节点和边所特征化。我们引入了一个算法框架（Het-node2vec），将原始的node2vec节点邻域采样方法扩展到了异构多图上。所得到的随机游走样本捕获了图的结构特征以及不同类型的节点和边的语义。所提出的算法可以聚焦于特定的节点或边类型，为所研究的预测问题中有兴趣的少数节点/边类型提供准确的表示。这些丰富而有针对性的表示可以增强对异构图的无监督和有监督学习。

    The development of Graph Representation Learning methods for heterogeneous graphs is fundamental in several real-world applications, since in several contexts graphs are characterized by different types of nodes and edges. We introduce a an algorithmic framework (Het-node2vec) that extends the original node2vec node-neighborhood sampling method to heterogeneous multigraphs. The resulting random walk samples capture both the structural characteristics of the graph and the semantics of the different types of nodes and edges. The proposed algorithms can focus their attention on specific node or edge types, allowing accurate representations also for underrepresented types of nodes/edges that are of interest for the prediction problem under investigation. These rich and well-focused representations can boost unsupervised and supervised learning on heterogeneous graphs.
    

