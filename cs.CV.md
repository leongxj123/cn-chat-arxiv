# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models.](http://arxiv.org/abs/2401.00873) | 该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。 |

# 详细

[^1]: 用贝叶斯方法统一自监督聚类和能量模型

    A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models. (arXiv:2401.00873v1 [cs.LG])

    [http://arxiv.org/abs/2401.00873](http://arxiv.org/abs/2401.00873)

    该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。

    

    自监督学习是一种利用大量无标签数据的流行且强大的方法，文献中提出了各种训练目标。本研究对最先进的自监督学习目标进行贝叶斯分析，阐明了每个类别中潜在的概率图模型，并提出了一种从基本原理出发推导这些模型的标准方法。分析还表明了将自监督学习与基于似然的生成模型自然整合的方法。我们在基于聚类的自监督学习和能量模型领域中实现了这个概念，引入了一个新的下界，经证明能可靠地惩罚最重要的失败模式。此外，这个新提出的下界使得能够训练一个标准的骨干架构，而无需使用诸如停止梯度、动量编码器或专门的聚类等非对称元素。

    Self-supervised learning is a popular and powerful method for utilizing large amounts of unlabeled data, for which a wide variety of training objectives have been proposed in the literature. In this study, we perform a Bayesian analysis of state-of-the-art self-supervised learning objectives, elucidating the underlying probabilistic graphical models in each class and presenting a standardized methodology for their derivation from first principles. The analysis also indicates a natural means of integrating self-supervised learning with likelihood-based generative models. We instantiate this concept within the realm of cluster-based self-supervised learning and energy models, introducing a novel lower bound which is proven to reliably penalize the most important failure modes. Furthermore, this newly proposed lower bound enables the training of a standard backbone architecture without the necessity for asymmetric elements such as stop gradients, momentum encoders, or specialized clusteri
    

