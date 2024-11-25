# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Huber-energy measure quantization.](http://arxiv.org/abs/2212.08162) | 该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。 |

# 详细

[^1]: Huber能量量化

    Huber-energy measure quantization. (arXiv:2212.08162v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.08162](http://arxiv.org/abs/2212.08162)

    该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。

    

    我们描述了一种测量量化过程，即一种算法，它通过$Q$个狄拉克函数的总和（$Q$为量化参数），找到目标概率定律（更一般地，为有限变差测度）的最佳逼近。该过程通过将原测度与其量化版本之间的统计距离最小化来实现；该距离基于负定核构建，并且如果必要，可以实时计算并输入随机优化算法（如SGD，Adam等）。我们在理论上研究了最优测量量化器的存在的基本问题，并确定了需要保证合适行为的核属性。我们提出了两个最佳线性无偏（BLUE）估计器，用于平方统计距离，并将它们用于无偏程序HEMQ中，以找到最佳量化。我们在多维高斯混合物、维纳空间魔方等几个数据库上测试了HEMQ

    We describe a measure quantization procedure i.e., an algorithm which finds the best approximation of a target probability law (and more generally signed finite variation measure) by a sum of $Q$ Dirac masses ($Q$ being the quantization parameter). The procedure is implemented by minimizing the statistical distance between the original measure and its quantized version; the distance is built from a negative definite kernel and, if necessary, can be computed on the fly and feed to a stochastic optimization algorithm (such as SGD, Adam, ...). We investigate theoretically the fundamental questions of existence of the optimal measure quantizer and identify what are the required kernel properties that guarantee suitable behavior. We propose two best linear unbiased (BLUE) estimators for the squared statistical distance and use them in an unbiased procedure, called HEMQ, to find the optimal quantization. We test HEMQ on several databases: multi-dimensional Gaussian mixtures, Wiener space cub
    

