# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation.](http://arxiv.org/abs/2308.02293) | 通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。 |

# 详细

[^1]: 用正则化高阶总变差的随机优化方法训练非线性神经网络

    A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation. (arXiv:2308.02293v1 [stat.ME])

    [http://arxiv.org/abs/2308.02293](http://arxiv.org/abs/2308.02293)

    通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。

    

    尽管包括深度神经网络在内的高度表达的参数模型可以更好地建模复杂概念，但训练这种高度非线性模型已知会导致严重的过拟合风险。针对这个问题，本研究考虑了一种k阶总变差（k-TV）正则化，它被定义为要训练的参数模型的k阶导数的平方积分，通过惩罚k-TV来产生一个更平滑的函数，从而避免过拟合。尽管将k-TV项应用于一般的参数模型由于积分而导致计算复杂，本研究提供了一种随机优化算法，可以高效地训练带有k-TV正则化的一般模型，而无需进行显式的数值积分。这种方法可以应用于结构任意的深度神经网络的训练，因为它只需要进行简单的随机梯度优化即可实现。

    While highly expressive parametric models including deep neural networks have an advantage to model complicated concepts, training such highly non-linear models is known to yield a high risk of notorious overfitting. To address this issue, this study considers a $k$th order total variation ($k$-TV) regularization, which is defined as the squared integral of the $k$th order derivative of the parametric models to be trained; penalizing the $k$-TV is expected to yield a smoother function, which is expected to avoid overfitting. While the $k$-TV terms applied to general parametric models are computationally intractable due to the integration, this study provides a stochastic optimization algorithm, that can efficiently train general models with the $k$-TV regularization without conducting explicit numerical integration. The proposed approach can be applied to the training of even deep neural networks whose structure is arbitrary, as it can be implemented by only a simple stochastic gradien
    

