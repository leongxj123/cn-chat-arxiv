# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Universal Lower Bounds and Optimal Rates: Achieving Minimax Clustering Error in Sub-Exponential Mixture Models](https://arxiv.org/abs/2402.15432) | 本文在混合模型中建立了一个通用下界，通过Chernoff散度来表达，将其拓展到具有次指数尾部的混合模型，并证明了迭代算法在这些混合模型中实现了最佳误差率 |

# 详细

[^1]: 在次指数混合模型中实现极小化聚类误差：通用下界和最佳速率

    Universal Lower Bounds and Optimal Rates: Achieving Minimax Clustering Error in Sub-Exponential Mixture Models

    [https://arxiv.org/abs/2402.15432](https://arxiv.org/abs/2402.15432)

    本文在混合模型中建立了一个通用下界，通过Chernoff散度来表达，将其拓展到具有次指数尾部的混合模型，并证明了迭代算法在这些混合模型中实现了最佳误差率

    

    聚类是无监督机器学习中的一个关键挑战，通常通过混合模型的视角来研究。在高斯和次高斯混合模型中恢复聚类标签的最佳误差率涉及到特定的信噪比。简单的迭代算法，如Lloyd算法，可以达到这个最佳误差率。在本文中，我们首先为任何混合模型中的误差率建立了一个通用下界，通过Chernoff散度来表达，这是一个比信噪比更通用的模型信息度量。然后我们证明了迭代算法在混合模型中实现了这个下界，特别强调了具有拉普拉斯分布误差的位置-尺度混合。此外，针对更适合由泊松或负二项混合模型建模的数据集，我们研究了其分布属于指数族的混合模型。

    arXiv:2402.15432v1 Announce Type: cross  Abstract: Clustering is a pivotal challenge in unsupervised machine learning and is often investigated through the lens of mixture models. The optimal error rate for recovering cluster labels in Gaussian and sub-Gaussian mixture models involves ad hoc signal-to-noise ratios. Simple iterative algorithms, such as Lloyd's algorithm, attain this optimal error rate. In this paper, we first establish a universal lower bound for the error rate in clustering any mixture model, expressed through a Chernoff divergence, a more versatile measure of model information than signal-to-noise ratios. We then demonstrate that iterative algorithms attain this lower bound in mixture models with sub-exponential tails, notably emphasizing location-scale mixtures featuring Laplace-distributed errors. Additionally, for datasets better modelled by Poisson or Negative Binomial mixtures, we study mixture models whose distributions belong to an exponential family. In such m
    

