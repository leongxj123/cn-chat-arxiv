# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Motif-aware Attribute Masking for Molecular Graph Pre-training.](http://arxiv.org/abs/2309.04589) | 本研究提出并研究了一种模式感知的属性屏蔽策略，通过利用相邻模式中的原子信息来捕捉模式间的结构，从而提高分子图预训练的效果。 |

# 详细

[^1]: 面向分子图预训练的模式感知属性屏蔽

    Motif-aware Attribute Masking for Molecular Graph Pre-training. (arXiv:2309.04589v1 [cs.LG])

    [http://arxiv.org/abs/2309.04589](http://arxiv.org/abs/2309.04589)

    本研究提出并研究了一种模式感知的属性屏蔽策略，通过利用相邻模式中的原子信息来捕捉模式间的结构，从而提高分子图预训练的效果。

    

    在图神经网络的预训练中，属性重构用于预测节点或边的特征。通过给定大量的分子，它们学习捕捉结构知识，这对于各种下游属性预测任务在化学、生物医学和材料科学中至关重要。先前的策略是随机选择节点进行属性屏蔽，利用局部邻居的信息。然而，对这些邻居的过度依赖抑制了模型从更高级的亚结构中学习。例如，模型从预测苯环中的三个碳原子中学到的信息很少，但是可以从功能基团之间的相互连接中学到更多信息，也可以称为化学模式。在这项工作中，我们提出并研究了模式感知的属性屏蔽策略，通过利用相邻模式中的原子信息来捕捉模式间的结构。一旦每个图被分解为不相交的

    Attribute reconstruction is used to predict node or edge features in the pre-training of graph neural networks. Given a large number of molecules, they learn to capture structural knowledge, which is transferable for various downstream property prediction tasks and vital in chemistry, biomedicine, and material science. Previous strategies that randomly select nodes to do attribute masking leverage the information of local neighbors However, the over-reliance of these neighbors inhibits the model's ability to learn from higher-level substructures. For example, the model would learn little from predicting three carbon atoms in a benzene ring based on the other three but could learn more from the inter-connections between the functional groups, or called chemical motifs. In this work, we propose and investigate motif-aware attribute masking strategies to capture inter-motif structures by leveraging the information of atoms in neighboring motifs. Once each graph is decomposed into disjoint
    

