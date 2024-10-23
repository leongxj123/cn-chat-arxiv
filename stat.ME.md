# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Fairness under Unobserved Confounding: A Neural Sensitivity Framework](https://arxiv.org/abs/2311.18460) | 分析了因果公平性对未观察到混杂的敏感性，推导出因果公平性指标的界限，提出神经框架用于学习公平预测，展示了框架的有效性 |
| [^2] | [Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec.](http://arxiv.org/abs/2310.17712) | 本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。 |

# 详细

[^1]: 未观察到的混杂下的因果公平性：一种神经敏感性框架

    Causal Fairness under Unobserved Confounding: A Neural Sensitivity Framework

    [https://arxiv.org/abs/2311.18460](https://arxiv.org/abs/2311.18460)

    分析了因果公平性对未观察到混杂的敏感性，推导出因果公平性指标的界限，提出神经框架用于学习公平预测，展示了框架的有效性

    

    机器学习预测中的公平性由于法律、道德和社会原因在实践中被广泛要求。现有工作通常集中在没有未观察到的混杂的设置上，尽管未观察到的混杂可能导致严重违反因果公平性，从而产生不公平的预测。在这项工作中，我们分析了因果公平性对未观察到的混杂的敏感性。我们的贡献有三个方面。首先，我们推导出不同来源的未观察到混杂下因果公平性指标的界限。这使从业者能够检查其机器学习模型对在公平关键应用中的未观察到的混杂的敏感性。其次，我们提出了一种用于学习公平预测的新型神经框架，这使我们能够提供对因果公平性可能由于未观察到的混杂而受到违反的程度的最坏情况保证。第三，我们展示了我们框架的有效性。

    arXiv:2311.18460v2 Announce Type: replace-cross  Abstract: Fairness for machine learning predictions is widely required in practice for legal, ethical, and societal reasons. Existing work typically focuses on settings without unobserved confounding, even though unobserved confounding can lead to severe violations of causal fairness and, thus, unfair predictions. In this work, we analyze the sensitivity of causal fairness to unobserved confounding. Our contributions are three-fold. First, we derive bounds for causal fairness metrics under different sources of unobserved confounding. This enables practitioners to examine the sensitivity of their machine learning models to unobserved confounding in fairness-critical applications. Second, we propose a novel neural framework for learning fair predictions, which allows us to offer worst-case guarantees of the extent to which causal fairness can be violated due to unobserved confounding. Third, we demonstrate the effectiveness of our framewor
    
[^2]: 使用Node2Vec学习到的嵌入进行社区检测和分类的保证

    Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec. (arXiv:2310.17712v1 [stat.ML])

    [http://arxiv.org/abs/2310.17712](http://arxiv.org/abs/2310.17712)

    本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。

    

    将大型网络的节点嵌入到欧几里得空间中是现代机器学习中的常见目标，有各种工具可用。这些嵌入可以用作社区检测/节点聚类或链接预测等任务的特征，其性能达到了最先进水平。除了谱聚类方法之外，对于其他常用的学习嵌入方法，缺乏理论上的理解。在这项工作中，我们考察了由node2vec学习到的嵌入的理论属性。我们的主要结果表明，对node2vec生成的嵌入向量应用k-means聚类可以对（经过度修正的）随机块模型中的节点进行弱一致的社区恢复。我们还讨论了这些嵌入在节点和链接预测任务中的应用。我们通过实验证明了这个结果，并研究了它与网络数据的其他嵌入工具之间的关系。

    Embedding the nodes of a large network into an Euclidean space is a common objective in modern machine learning, with a variety of tools available. These embeddings can then be used as features for tasks such as community detection/node clustering or link prediction, where they achieve state of the art performance. With the exception of spectral clustering methods, there is little theoretical understanding for other commonly used approaches to learning embeddings. In this work we examine the theoretical properties of the embeddings learned by node2vec. Our main result shows that the use of k-means clustering on the embedding vectors produced by node2vec gives weakly consistent community recovery for the nodes in (degree corrected) stochastic block models. We also discuss the use of these embeddings for node and link prediction tasks. We demonstrate this result empirically, and examine how this relates to other embedding tools for network data.
    

