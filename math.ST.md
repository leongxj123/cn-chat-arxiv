# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarially robust clustering with optimality guarantees.](http://arxiv.org/abs/2306.09977) | 本文提出了一种简单的算法，即使在存在对抗性的异常值的情况下，也能获得最优的错标率。在没有异常值的情况下，该算法能够实现与洛伊德算法类似的理论保证. |

# 详细

[^1]: 带有最优性保证的对抗鲁棒聚类

    Adversarially robust clustering with optimality guarantees. (arXiv:2306.09977v1 [math.ST])

    [http://arxiv.org/abs/2306.09977](http://arxiv.org/abs/2306.09977)

    本文提出了一种简单的算法，即使在存在对抗性的异常值的情况下，也能获得最优的错标率。在没有异常值的情况下，该算法能够实现与洛伊德算法类似的理论保证.

    

    我们考虑对来自亚高斯混合的数据点进行聚类的问题。现有的可证明达到最优错标率的方法，如洛伊德算法，通常容易受到异常值的影响。相反，似乎对对抗性扰动具有鲁棒性的聚类方法不知道是否满足最优的统计保证。我们提出了一种简单的算法，即使允许出现对抗性的异常值，也能获得最优的错标率。当满足弱初始化条件时，我们的算法在常数次迭代中实现最优误差率。在没有异常值的情况下，在固定维度上，我们的理论保证与洛伊德算法类似。在各种模拟数据集上进行了广泛的实验，以支持我们的方法的理论保证。

    We consider the problem of clustering data points coming from sub-Gaussian mixtures. Existing methods that provably achieve the optimal mislabeling error, such as the Lloyd algorithm, are usually vulnerable to outliers. In contrast, clustering methods seemingly robust to adversarial perturbations are not known to satisfy the optimal statistical guarantees. We propose a simple algorithm that obtains the optimal mislabeling rate even when we allow adversarial outliers to be present. Our algorithm achieves the optimal error rate in constant iterations when a weak initialization condition is satisfied. In the absence of outliers, in fixed dimensions, our theoretical guarantees are similar to that of the Lloyd algorithm. Extensive experiments on various simulated data sets are conducted to support the theoretical guarantees of our method.
    

