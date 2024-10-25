# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tail-adaptive Bayesian shrinkage](https://arxiv.org/abs/2007.02192) | 提出了一种在多样的稀疏情况下具有尾部自适应收缩特性的鲁棒稀疏估计方法，通过新的全局-局部-尾部高斯混合分布实现，能够根据稀疏程度自适应调整先验的尾部重量以适应更多或更少信号。 |

# 详细

[^1]: 尾部自适应贝叶斯收缩

    Tail-adaptive Bayesian shrinkage

    [https://arxiv.org/abs/2007.02192](https://arxiv.org/abs/2007.02192)

    提出了一种在多样的稀疏情况下具有尾部自适应收缩特性的鲁棒稀疏估计方法，通过新的全局-局部-尾部高斯混合分布实现，能够根据稀疏程度自适应调整先验的尾部重量以适应更多或更少信号。

    

    本文研究了高维回归问题下多样的稀疏情况下的鲁棒贝叶斯方法。传统的收缩先验主要设计用于在所谓的超稀疏领域从成千上万个预测变量中检测少数信号。然而，当稀疏程度适中时，它们可能表现不尽人意。在本文中，我们提出了一种在多样稀疏情况下具有尾部自适应收缩特性的鲁棒稀疏估计方法。在这种特性中，先验的尾部重量会自适应调整，随着稀疏水平的增加或减少变得更大或更小，以适应先验地更多或更少的信号。我们提出了一个全局局部尾部（GLT）高斯混合分布以确保这种属性。我们考察了先验的尾部指数与基础稀疏水平之间的关系，并证明GLT后验会在...

    arXiv:2007.02192v4 Announce Type: replace-cross  Abstract: Robust Bayesian methods for high-dimensional regression problems under diverse sparse regimes are studied. Traditional shrinkage priors are primarily designed to detect a handful of signals from tens of thousands of predictors in the so-called ultra-sparsity domain. However, they may not perform desirably when the degree of sparsity is moderate. In this paper, we propose a robust sparse estimation method under diverse sparsity regimes, which has a tail-adaptive shrinkage property. In this property, the tail-heaviness of the prior adjusts adaptively, becoming larger or smaller as the sparsity level increases or decreases, respectively, to accommodate more or fewer signals, a posteriori. We propose a global-local-tail (GLT) Gaussian mixture distribution that ensures this property. We examine the role of the tail-index of the prior in relation to the underlying sparsity level and demonstrate that the GLT posterior contracts at the
    

