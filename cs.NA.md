# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Numerical Stability of Hyperbolic Representation Learning.](http://arxiv.org/abs/2211.00181) | 本文研究了超几何表征学习中的数值不稳定性问题，比较了两种流行的超几何模型Poincar\'e球和Lorentz模型，发现Lorentz模型具有更好的数值稳定性和优化性能，同时提出一种新的欧几里得优化方案作为超几何学习的另一个选择。 |

# 详细

[^1]: 超几何表征学习的数值稳定性

    The Numerical Stability of Hyperbolic Representation Learning. (arXiv:2211.00181v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.00181](http://arxiv.org/abs/2211.00181)

    本文研究了超几何表征学习中的数值不稳定性问题，比较了两种流行的超几何模型Poincar\'e球和Lorentz模型，发现Lorentz模型具有更好的数值稳定性和优化性能，同时提出一种新的欧几里得优化方案作为超几何学习的另一个选择。

    

    由于超球的容量随半径的指数增长，超几何空间能够将具有层次结构的数据集嵌入其中而不失真。然而，这种指数增长的性质常常导致数值不稳定性，使得训练超几何学习模型有时会导致灾难性的NaN问题和浮点算术中遇到无法表示的值。在本文中，我们对两种广泛使用的超几何模型——Poincar\'e球和Lorentz模型的局限性进行了仔细的分析。我们首先展示了，在64位算术系统下，Poincar\'e球相对于Lorentz模型具有更大的能力来正确表示点。然后，我们从优化的角度理论上验证了Lorentz模型优于Poincar\'e球的优越性。鉴于两种模型的数值限制，我们确定一种欧几里得优化方案，在Poincar\'e球和Lorentz模型之外为超几何学习提供了一种新的方案。

    Given the exponential growth of the volume of the ball w.r.t. its radius, the hyperbolic space is capable of embedding trees with arbitrarily small distortion and hence has received wide attention for representing hierarchical datasets. However, this exponential growth property comes at a price of numerical instability such that training hyperbolic learning models will sometimes lead to catastrophic NaN problems, encountering unrepresentable values in floating point arithmetic. In this work, we carefully analyze the limitation of two popular models for the hyperbolic space, namely, the Poincar\'e ball and the Lorentz model. We first show that, under the 64 bit arithmetic system, the Poincar\'e ball has a relatively larger capacity than the Lorentz model for correctly representing points. Then, we theoretically validate the superiority of the Lorentz model over the Poincar\'e ball from the perspective of optimization. Given the numerical limitations of both models, we identify one Eucli
    

