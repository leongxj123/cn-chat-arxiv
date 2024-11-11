# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph neural network outputs are almost surely asymptotically constant](https://arxiv.org/abs/2403.03880) | 研究表明，图神经网络的输出将渐近于一个常数函数，并限制了这些分类器的统一表达能力。 |

# 详细

[^1]: 图神经网络的输出几乎肯定是渐近常数

    Graph neural network outputs are almost surely asymptotically constant

    [https://arxiv.org/abs/2403.03880](https://arxiv.org/abs/2403.03880)

    研究表明，图神经网络的输出将渐近于一个常数函数，并限制了这些分类器的统一表达能力。

    

    图神经网络（GNNs）是各种图学习任务中主要的架构。我们通过研究GNN的概率分类器在从某个随机图模型中绘制的更大图上应用时预测如何演变，提出了GNN表达能力的新角度。我们展示了输出收敛到一个常数函数，这个函数上限了这些分类器可以统一表达的内容。这种收敛现象适用于非常广泛的GNN类别，包括先进模型，其中的聚合包括平均值和基于注意力的图转换器机制。我们的结果适用于各种随机图模型，包括（稀疏的）Erd\H{o}s-R\'enyi模型和随机块模型。我们通过实证验证这些发现，观察到收敛现象已经在相对适中规模的图中显现。

    arXiv:2403.03880v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are the predominant architectures for a variety of learning tasks on graphs. We present a new angle on the expressive power of GNNs by studying how the predictions of a GNN probabilistic classifier evolve as we apply it on larger graphs drawn from some random graph model. We show that the output converges to a constant function, which upper-bounds what these classifiers can express uniformly. This convergence phenomenon applies to a very wide class of GNNs, including state of the art models, with aggregates including mean and the attention-based mechanism of graph transformers. Our results apply to a broad class of random graph models, including the (sparse) Erd\H{o}s-R\'enyi model and the stochastic block model. We empirically validate these findings, observing that the convergence phenomenon already manifests itself on graphs of relatively modest size.
    

