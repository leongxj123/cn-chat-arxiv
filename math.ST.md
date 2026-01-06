# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Geometry-induced Implicit Regularization in Deep ReLU Neural Networks](https://arxiv.org/abs/2402.08269) | 通过研究参数变化时输出集合的几何特征，我们发现在深度ReLU神经网络的优化过程中存在几何引导的隐式正则化现象。 |
| [^2] | [Sample Path Regularity of Gaussian Processes from the Covariance Kernel](https://arxiv.org/abs/2312.14886) | 本文提供了关于高斯过程样本路径正则性的新颖和紧凑的特征描述，通过协方差核对应的GP样本路径达到一定正则性的充分必要条件，对常用于机器学习应用中的GPs的样本路径正则性进行了探讨。 |

# 详细

[^1]: 深度ReLU神经网络中的几何引导隐式正则化

    Geometry-induced Implicit Regularization in Deep ReLU Neural Networks

    [https://arxiv.org/abs/2402.08269](https://arxiv.org/abs/2402.08269)

    通过研究参数变化时输出集合的几何特征，我们发现在深度ReLU神经网络的优化过程中存在几何引导的隐式正则化现象。

    

    众所周知，具有比训练样本更多参数的神经网络不会过拟合。隐式正则化现象在优化过程中出现，对“好”的网络有利。因此，如果我们不考虑所有可能的网络，而只考虑“好”的网络，参数数量就不是一个足够衡量复杂性的指标。为了更好地理解在优化过程中哪些网络受到青睐，我们研究了参数变化时输出集合的几何特征。当输入固定时，我们证明了这个集合的维度会发生变化，并且局部维度，即批次功能维度，几乎总是由隐藏层中的激活模式决定。我们证明了批次功能维度对网络参数化的对称性（神经元排列和正向缩放）是不变的。实证上，我们证实了在优化过程中批次功能维度会下降。因此，优化过程具有隐式正则化的效果。

    It is well known that neural networks with many more parameters than training examples do not overfit. Implicit regularization phenomena, which are still not well understood, occur during optimization and 'good' networks are favored. Thus the number of parameters is not an adequate measure of complexity if we do not consider all possible networks but only the 'good' ones. To better understand which networks are favored during optimization, we study the geometry of the output set as parameters vary. When the inputs are fixed, we prove that the dimension of this set changes and that the local dimension, called batch functional dimension, is almost surely determined by the activation patterns in the hidden layers. We prove that the batch functional dimension is invariant to the symmetries of the network parameterization: neuron permutations and positive rescalings. Empirically, we establish that the batch functional dimension decreases during optimization. As a consequence, optimization l
    
[^2]: 来自协方差核的高斯过程样本路径正则性

    Sample Path Regularity of Gaussian Processes from the Covariance Kernel

    [https://arxiv.org/abs/2312.14886](https://arxiv.org/abs/2312.14886)

    本文提供了关于高斯过程样本路径正则性的新颖和紧凑的特征描述，通过协方差核对应的GP样本路径达到一定正则性的充分必要条件，对常用于机器学习应用中的GPs的样本路径正则性进行了探讨。

    

    高斯过程（GPs）是定义函数空间上的概率分布的最常见形式主义。尽管GPs的应用广泛，但对于GP样本路径的全面理解，即它们定义概率测度的函数空间，尚缺乏。在实践中，GPs不是通过概率测度构建的，而是通过均值函数和协方差核构建的。本文针对协方差核提供了GP样本路径达到给定正则性所需的充分必要条件。我们使用H\"older正则性框架，因为它提供了特别简单的条件，在平稳和各向同性GPs的情况下进一步简化。然后，我们证明我们的结果允许对机器学习应用中常用的GPs的样本路径正则性进行新颖且异常紧凑的表征。

    arXiv:2312.14886v2 Announce Type: replace  Abstract: Gaussian processes (GPs) are the most common formalism for defining probability distributions over spaces of functions. While applications of GPs are myriad, a comprehensive understanding of GP sample paths, i.e. the function spaces over which they define a probability measure, is lacking. In practice, GPs are not constructed through a probability measure, but instead through a mean function and a covariance kernel. In this paper we provide necessary and sufficient conditions on the covariance kernel for the sample paths of the corresponding GP to attain a given regularity. We use the framework of H\"older regularity as it grants particularly straightforward conditions, which simplify further in the cases of stationary and isotropic GPs. We then demonstrate that our results allow for novel and unusually tight characterisations of the sample path regularities of the GPs commonly used in machine learning applications, such as the Mat\'
    

