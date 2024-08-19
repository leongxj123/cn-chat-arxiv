# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Out-of-Distribution Generalization via Causal Intervention](https://arxiv.org/abs/2402.11494) | GNN在离群分布泛化中的失败关键在于来自环境的潜在混杂偏差，因此引入了一个简单而原则性的方法来训练稳健GNN。 |
| [^2] | [Simplifying and Empowering Transformers for Large-Graph Representations.](http://arxiv.org/abs/2306.10759) | 本文通过实验证明，在大型图上使用一层注意力即可获得令人惊讶的竞争性能，挑战了在语言和视觉任务中复杂模型的应用。这促使我们重新思考在大型图上设计Transformer的理念，以提高可扩展性。 |

# 详细

[^1]: 通过因果干预实现图形的离群分布泛化

    Graph Out-of-Distribution Generalization via Causal Intervention

    [https://arxiv.org/abs/2402.11494](https://arxiv.org/abs/2402.11494)

    GNN在离群分布泛化中的失败关键在于来自环境的潜在混杂偏差，因此引入了一个简单而原则性的方法来训练稳健GNN。

    

    离群分布（OOD）泛化在图形学习中引起了越来越多的关注，因为图神经网络（GNN）在分布转移时通常会表现出性能下降。本文从自下而上的数据生成角度出发，通过因果分析揭示了一个关键观察：GNN在OOD泛化中失败的关键在于来自环境的潜在混杂偏差。后者误导模型利用自我图特征和目标节点标签之间的环境敏感相关性，导致在新的未见节点上不良泛化。基于这一分析，我们引入了一个在节点级别分布转移下训练稳健GNN的概念简单而又原则性的方法，而不需要环境的先验知识。

    arXiv:2402.11494v1 Announce Type: new  Abstract: Out-of-distribution (OOD) generalization has gained increasing attentions for learning on graphs, as graph neural networks (GNNs) often exhibit performance degradation with distribution shifts. The challenge is that distribution shifts on graphs involve intricate interconnections between nodes, and the environment labels are often absent in data. In this paper, we adopt a bottom-up data-generative perspective and reveal a key observation through causal analysis: the crux of GNNs' failure in OOD generalization lies in the latent confounding bias from the environment. The latter misguides the model to leverage environment-sensitive correlations between ego-graph features and target nodes' labels, resulting in undesirable generalization on new unseen nodes. Built upon this analysis, we introduce a conceptually simple yet principled approach for training robust GNNs under node-level distribution shifts, without prior knowledge of environment
    
[^2]: 为大型图表示简化和增强Transformer

    Simplifying and Empowering Transformers for Large-Graph Representations. (arXiv:2306.10759v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.10759](http://arxiv.org/abs/2306.10759)

    本文通过实验证明，在大型图上使用一层注意力即可获得令人惊讶的竞争性能，挑战了在语言和视觉任务中复杂模型的应用。这促使我们重新思考在大型图上设计Transformer的理念，以提高可扩展性。

    

    在大型图上学习表示是一个长期存在的挑战，因为其中涉及了大量数据点之间的相互依赖关系。Transformer作为一种新兴的用于图结构数据的基本编码器类别，由于其全局注意力可以捕捉到邻节点之外的所有对影响，因此在小型图上表现出了有希望的性能。尽管如此，现有方法往往继承了Transformer在语言和视觉任务中的思想，并通过堆叠深层多头注意力来采用复杂的模型。本文通过关于节点属性预测基准的实验证明，即使只使用一层注意力也能在节点数量从千级到十亿级的范围内带来令人惊讶的竞争性能。这鼓励我们重新思考在大型图上设计Transformer的理念，其中全局注意力是一个阻碍可扩展性的计算开销。我们将提出的方案称为简化图Transformer。

    Learning representations on large-sized graphs is a long-standing challenge due to the inter-dependence nature involved in massive data points. Transformers, as an emerging class of foundation encoders for graph-structured data, have shown promising performance on small graphs due to its global attention capable of capturing all-pair influence beyond neighboring nodes. Even so, existing approaches tend to inherit the spirit of Transformers in language and vision tasks, and embrace complicated models by stacking deep multi-head attentions. In this paper, we critically demonstrate that even using a one-layer attention can bring up surprisingly competitive performance across node property prediction benchmarks where node numbers range from thousand-level to billion-level. This encourages us to rethink the design philosophy for Transformers on large graphs, where the global attention is a computation overhead hindering the scalability. We frame the proposed scheme as Simplified Graph Trans
    

