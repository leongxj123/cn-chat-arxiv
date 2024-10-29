# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference.](http://arxiv.org/abs/2401.01426) | 本文提出了一种用于高维因果推断的模块化深度生成模型学习算法，该算法利用预训练的模型来回答由高维数据引起的因果查询。 |

# 详细

[^1]: 高维因果推断的模块化深度生成模型学习

    Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference. (arXiv:2401.01426v1 [cs.LG])

    [http://arxiv.org/abs/2401.01426](http://arxiv.org/abs/2401.01426)

    本文提出了一种用于高维因果推断的模块化深度生成模型学习算法，该算法利用预训练的模型来回答由高维数据引起的因果查询。

    

    Pearl的因果层次结构在观测、干预和反事实问题之间建立了明确的分离。研究人员提出了计算可辨识因果查询的声音和完整算法，在给定层次的因果结构和数据的情况下使用较低层次的层次的数据。然而，大多数这些算法假设我们可以准确估计数据的概率分布，这对于如图像这样的高维变量是一个不切实际的假设。另一方面，现代生成式深度学习架构可以被训练来学习如何准确地从这样的高维分布中采样。特别是随着图像基模型的最近兴起，利用预训练模型来回答带有这样高维数据的因果查询是非常有吸引力的。为了解决这个问题，我们提出了一个顺序训练算法，给定因果结构和预训练的条件生成模型，可以训练一个模型来估计由高维数据引起的因果关系。

    Pearl's causal hierarchy establishes a clear separation between observational, interventional, and counterfactual questions. Researchers proposed sound and complete algorithms to compute identifiable causal queries at a given level of the hierarchy using the causal structure and data from the lower levels of the hierarchy. However, most of these algorithms assume that we can accurately estimate the probability distribution of the data, which is an impractical assumption for high-dimensional variables such as images. On the other hand, modern generative deep learning architectures can be trained to learn how to accurately sample from such high-dimensional distributions. Especially with the recent rise of foundation models for images, it is desirable to leverage pre-trained models to answer causal queries with such high-dimensional data. To address this, we propose a sequential training algorithm that, given the causal structure and a pre-trained conditional generative model, can train a
    

