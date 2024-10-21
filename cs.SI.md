# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Path-based Explanation for Knowledge Graph Completion.](http://arxiv.org/abs/2401.02290) | 基于路径的KGC解释器Power-Link通过引入图加权技术，实现了可解释的知识图谱补全，推动了模型透明度和可靠性的提升。 |
| [^2] | [Simplifying Subgraph Representation Learning for Scalable Link Prediction.](http://arxiv.org/abs/2301.12562) | 提出了一种新的可扩展简化子图表示学习（S3GRL）框架，通过简化每个链接子图中的消息传递和聚合操作实现更快的训练和推理，并可适应各种子图采样策略和扩散操作符以模拟计算代价高的子图表示学习。大量实验证明了S3GRL模型可以扩展SGRL而不会显著降低性能。 |

# 详细

[^1]: 基于路径的知识图谱补全的解释方法

    Path-based Explanation for Knowledge Graph Completion. (arXiv:2401.02290v1 [cs.LG])

    [http://arxiv.org/abs/2401.02290](http://arxiv.org/abs/2401.02290)

    基于路径的KGC解释器Power-Link通过引入图加权技术，实现了可解释的知识图谱补全，推动了模型透明度和可靠性的提升。

    

    近年来，图神经网络（GNNs）通过建模实体和关系的交互在知识图谱补全（KGC）任务中取得了巨大成功。然而，对预测结果的解释却没有得到必要的关注。对基于GNN的KGC模型结果进行适当解释，可以增加模型的透明度，并帮助研究人员开发更可靠的模型。现有的KGC解释方法主要依赖于实例/子图的方法，而在某些场景下，路径可以提供更友好和可解释的解释。然而，还没有对生成基于路径的知识图谱解释方法进行充分探索。为了填补这一空白，我们提出了Power-Link，这是第一个探索基于路径的KGC解释器。我们设计了一种新颖的图加权技术，使得可以以完全可并行化和内存高效的训练方案生成基于路径的解释。我们还引入了三个新的度量指标，用于评估解释的质量和有效性。

    Graph Neural Networks (GNNs) have achieved great success in Knowledge Graph Completion (KGC) by modelling how entities and relations interact in recent years. However, the explanation of the predicted facts has not caught the necessary attention. Proper explanations for the results of GNN-based KGC models increase model transparency and help researchers develop more reliable models. Existing practices for explaining KGC tasks rely on instance/subgraph-based approaches, while in some scenarios, paths can provide more user-friendly and interpretable explanations. Nonetheless, the methods for generating path-based explanations for KGs have not been well-explored. To address this gap, we propose Power-Link, the first path-based KGC explainer that explores GNN-based models. We design a novel simplified graph-powering technique, which enables the generation of path-based explanations with a fully parallelisable and memory-efficient training scheme. We further introduce three new metrics for 
    
[^2]: 针对可扩展性的子图表示学习简化以进行可扩展的链接预测

    Simplifying Subgraph Representation Learning for Scalable Link Prediction. (arXiv:2301.12562v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12562](http://arxiv.org/abs/2301.12562)

    提出了一种新的可扩展简化子图表示学习（S3GRL）框架，通过简化每个链接子图中的消息传递和聚合操作实现更快的训练和推理，并可适应各种子图采样策略和扩散操作符以模拟计算代价高的子图表示学习。大量实验证明了S3GRL模型可以扩展SGRL而不会显著降低性能。

    

    图上的链接预测是一个基本问题。子图表示学习方法通过将链接预测转化为在链接周围子图上的图分类来实现了最先进的链接预测。然而，子图表示学习方法计算代价高，并且由于子图水平操作的代价而不适用于大规模图形。我们提出了一种新的子图表示学习类，称为可扩展简化子图表示学习（S3GRL），旨在实现更快的训练和推理。S3GRL简化了每个链接子图中的消息传递和聚合操作。作为可扩展性框架，S3GRL适应各种子图采样策略和扩散运算符来模拟计算代价高的子图表示学习方法。我们提出了多个S3GRL实例，并在小到大规模的图形上进行了实证研究。我们广泛的实验表明，所提出的S3GRL模型可以扩展SGRL而不会显著降低性能。

    Link prediction on graphs is a fundamental problem. Subgraph representation learning approaches (SGRLs), by transforming link prediction to graph classification on the subgraphs around the links, have achieved state-of-the-art performance in link prediction. However, SGRLs are computationally expensive, and not scalable to large-scale graphs due to expensive subgraph-level operations. To unlock the scalability of SGRLs, we propose a new class of SGRLs, that we call Scalable Simplified SGRL (S3GRL). Aimed at faster training and inference, S3GRL simplifies the message passing and aggregation operations in each link's subgraph. S3GRL, as a scalability framework, accommodates various subgraph sampling strategies and diffusion operators to emulate computationally-expensive SGRLs. We propose multiple instances of S3GRL and empirically study them on small to large-scale graphs. Our extensive experiments demonstrate that the proposed S3GRL models scale up SGRLs without significant performance 
    

