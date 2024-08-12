# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Domain Generalization through Meta-Learning: A Survey](https://arxiv.org/abs/2404.02785) | 元学习是一种有前景的方法，通过获取可转移知识实现在各种任务之间快速适应，为解决深度神经网络在面对分布变化和有限标记数据时泛化能力不佳提供了新途径。 |
| [^2] | [Sensitivity-Aware Mixed-Precision Quantization and Width Optimization of Deep Neural Networks Through Cluster-Based Tree-Structured Parzen Estimation.](http://arxiv.org/abs/2308.06422) | 本研究引入了一种创新的深度神经网络优化方法，通过自动选择最佳的位宽和层宽来提高网络效率。同时，通过剪枝和聚类技术，优化了搜索过程，并在多个数据集上进行了严格测试，结果显示该方法明显优于现有方法。 |

# 详细

[^1]: 通过元学习实现领域泛化：一项调查

    Domain Generalization through Meta-Learning: A Survey

    [https://arxiv.org/abs/2404.02785](https://arxiv.org/abs/2404.02785)

    元学习是一种有前景的方法，通过获取可转移知识实现在各种任务之间快速适应，为解决深度神经网络在面对分布变化和有限标记数据时泛化能力不佳提供了新途径。

    

    深度神经网络(DNNs)已经彻底改变了人工智能，但是当面对分布之外(out-of-distribution, OOD)数据时往往表现不佳，这是因为在现实世界应用中由于领域转移不可避免，训练和测试数据被假定为共享相同分布的常见情况。尽管DNNs在大量数据和计算能力方面非常有效，但它们很难应对分布变化和有限标记数据，导致过拟合和跨不同任务和领域的泛化能力不佳。元学习提供了一种有前途的方法，通过采用能够在各种任务之间获取可转移知识的算法进行快速适应，从而消除了需要从头学习每个任务的必要性。本调查论文深入探讨了元学习领域，重点关注其对领域泛化的贡献。

    arXiv:2404.02785v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) have revolutionized artificial intelligence but often lack performance when faced with out-of-distribution (OOD) data, a common scenario due to the inevitable domain shifts in real-world applications. This limitation stems from the common assumption that training and testing data share the same distribution-an assumption frequently violated in practice. Despite their effectiveness with large amounts of data and computational power, DNNs struggle with distributional shifts and limited labeled data, leading to overfitting and poor generalization across various tasks and domains. Meta-learning presents a promising approach by employing algorithms that acquire transferable knowledge across various tasks for fast adaptation, eliminating the need to learn each task from scratch. This survey paper delves into the realm of meta-learning with a focus on its contribution to domain generalization. We first clarify the 
    
[^2]: 使用基于聚类的树状Parzen估计的敏感性感知混合精度量化和宽度优化的深度神经网络

    Sensitivity-Aware Mixed-Precision Quantization and Width Optimization of Deep Neural Networks Through Cluster-Based Tree-Structured Parzen Estimation. (arXiv:2308.06422v1 [cs.LG])

    [http://arxiv.org/abs/2308.06422](http://arxiv.org/abs/2308.06422)

    本研究引入了一种创新的深度神经网络优化方法，通过自动选择最佳的位宽和层宽来提高网络效率。同时，通过剪枝和聚类技术，优化了搜索过程，并在多个数据集上进行了严格测试，结果显示该方法明显优于现有方法。

    

    随着深度学习模型的复杂性和计算需求的提高，对神经网络设计的有效优化方法的需求变得至关重要。本文引入了一种创新的搜索机制，用于自动选择单个神经网络层的最佳位宽和层宽。这导致深度神经网络效率的明显提高。通过利用基于Hessian的剪枝策略，有选择地减少搜索域，确保移除非关键参数。随后，我们通过使用基于聚类的树状Parzen估计器开发有利和不利结果的替代模型。这种策略允许对架构可能性进行简化的探索，并迅速确定表现最好的设计。通过对知名数据集进行严格测试，我们的方法证明了与现有方法相比的明显优势。与领先的压缩策略相比，我们的方法取得了令人瞩目的成果。

    As the complexity and computational demands of deep learning models rise, the need for effective optimization methods for neural network designs becomes paramount. This work introduces an innovative search mechanism for automatically selecting the best bit-width and layer-width for individual neural network layers. This leads to a marked enhancement in deep neural network efficiency. The search domain is strategically reduced by leveraging Hessian-based pruning, ensuring the removal of non-crucial parameters. Subsequently, we detail the development of surrogate models for favorable and unfavorable outcomes by employing a cluster-based tree-structured Parzen estimator. This strategy allows for a streamlined exploration of architectural possibilities and swift pinpointing of top-performing designs. Through rigorous testing on well-known datasets, our method proves its distinct advantage over existing methods. Compared to leading compression strategies, our approach records an impressive 
    

