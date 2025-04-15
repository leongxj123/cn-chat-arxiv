# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective](https://arxiv.org/abs/2402.04621) | A-X依赖关系是影响图卷积效果的重要因素，特征重排可以显著提升图神经网络的性能。 |

# 详细

[^1]: 图拓扑结构上的特征分布调节了图卷积的效果：同质性视角

    Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective

    [https://arxiv.org/abs/2402.04621](https://arxiv.org/abs/2402.04621)

    A-X依赖关系是影响图卷积效果的重要因素，特征重排可以显著提升图神经网络的性能。

    

    随机重排同一类别节点之间的特征向量如何影响图神经网络（GNNs）？直观地说，特征重排扰乱了GNNs从图拓扑和特征之间的依赖关系（A-X依赖关系），从而影响了GNNs的学习。令人惊讶的是，在特征重排之后，我们观察到GNNs的性能显著提升。由于忽视了A-X依赖关系对GNNs的影响，先前的文献没有给出对该现象的满意解释。因此，我们提出了两个研究问题。首先，如何在控制潜在混淆因素的情况下度量A-X依赖关系？其次，A-X依赖关系如何影响GNNs？作为回应，我们（i）提出了一种基于原则的度量A-X依赖关系的方法，（ii）设计了一个控制A-X依赖关系的随机图模型，（iii）建立了A-X依赖关系与图卷积之间关系的理论，以及（iv）对实际图进行了与理论一致的实证分析。我们认为A-X依赖关系对GNNs具有重要影响。

    How would randomly shuffling feature vectors among nodes from the same class affect graph neural networks (GNNs)? The feature shuffle, intuitively, perturbs the dependence between graph topology and features (A-X dependence) for GNNs to learn from. Surprisingly, we observe a consistent and significant improvement in GNN performance following the feature shuffle. Having overlooked the impact of A-X dependence on GNNs, the prior literature does not provide a satisfactory understanding of the phenomenon. Thus, we raise two research questions. First, how should A-X dependence be measured, while controlling for potential confounds? Second, how does A-X dependence affect GNNs? In response, we (i) propose a principled measure for A-X dependence, (ii) design a random graph model that controls A-X dependence, (iii) establish a theory on how A-X dependence relates to graph convolution, and (iv) present empirical analysis on real-world graphs that aligns with the theory. We conclude that A-X depe
    

