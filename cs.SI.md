# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec.](http://arxiv.org/abs/2310.17712) | 本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。 |

# 详细

[^1]: 使用Node2Vec学习到的嵌入进行社区检测和分类的保证

    Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec. (arXiv:2310.17712v1 [stat.ML])

    [http://arxiv.org/abs/2310.17712](http://arxiv.org/abs/2310.17712)

    本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。

    

    将大型网络的节点嵌入到欧几里得空间中是现代机器学习中的常见目标，有各种工具可用。这些嵌入可以用作社区检测/节点聚类或链接预测等任务的特征，其性能达到了最先进水平。除了谱聚类方法之外，对于其他常用的学习嵌入方法，缺乏理论上的理解。在这项工作中，我们考察了由node2vec学习到的嵌入的理论属性。我们的主要结果表明，对node2vec生成的嵌入向量应用k-means聚类可以对（经过度修正的）随机块模型中的节点进行弱一致的社区恢复。我们还讨论了这些嵌入在节点和链接预测任务中的应用。我们通过实验证明了这个结果，并研究了它与网络数据的其他嵌入工具之间的关系。

    Embedding the nodes of a large network into an Euclidean space is a common objective in modern machine learning, with a variety of tools available. These embeddings can then be used as features for tasks such as community detection/node clustering or link prediction, where they achieve state of the art performance. With the exception of spectral clustering methods, there is little theoretical understanding for other commonly used approaches to learning embeddings. In this work we examine the theoretical properties of the embeddings learned by node2vec. Our main result shows that the use of k-means clustering on the embedding vectors produced by node2vec gives weakly consistent community recovery for the nodes in (degree corrected) stochastic block models. We also discuss the use of these embeddings for node and link prediction tasks. We demonstrate this result empirically, and examine how this relates to other embedding tools for network data.
    

