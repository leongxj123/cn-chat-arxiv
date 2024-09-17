# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving the Adaptive Moment Estimation (ADAM) stochastic optimizer through an Implicit-Explicit (IMEX) time-stepping approach](https://arxiv.org/abs/2403.13704) | 通过隐式显式(IMEX)时间步进方法改进自适应矩估计（ADAM）随机优化器，提出了一种新的神经网络训练优化算法，比经典Adam在几个回归和分类问题上表现更好。 |
| [^2] | [Approximation by non-symmetric networks for cross-domain learning.](http://arxiv.org/abs/2305.03890) | 本文研究使用非对称内核进行基于内核网络逼近的通用方法，结果表明它可以在跨域学习中显著提高基于内核网络的逼近能力。 |

# 详细

[^1]: 通过隐式显式(IMEX)时间步进方法改进自适应矩估计（ADAM）随机优化器

    Improving the Adaptive Moment Estimation (ADAM) stochastic optimizer through an Implicit-Explicit (IMEX) time-stepping approach

    [https://arxiv.org/abs/2403.13704](https://arxiv.org/abs/2403.13704)

    通过隐式显式(IMEX)时间步进方法改进自适应矩估计（ADAM）随机优化器，提出了一种新的神经网络训练优化算法，比经典Adam在几个回归和分类问题上表现更好。

    

    Adam优化器通常用于神经网络训练中，对应于在非常小的学习速率限制下的基本常微分方程（ODE）。本文表明，经典Adam算法是底层ODE的一阶隐式显式(IMEX) Euler离散化。从时间离散化角度出发，我们提出了通过使用更高阶IMEX方法来解决ODE的Adam方案的新扩展。基于这种方法，我们推导了一种新的神经网络训练优化算法，在几个回归和分类问题上比经典Adam表现更好。

    arXiv:2403.13704v1 Announce Type: cross  Abstract: The Adam optimizer, often used in Machine Learning for neural network training, corresponds to an underlying ordinary differential equation (ODE) in the limit of very small learning rates. This work shows that the classical Adam algorithm is a first order implicit-explicit (IMEX) Euler discretization of the underlying ODE. Employing the time discretization point of view, we propose new extensions of the Adam scheme obtained by using higher order IMEX methods to solve the ODE. Based on this approach, we derive a new optimization algorithm for neural network training that performs better than classical Adam on several regression and classification problems.
    
[^2]: 非对称网络逼近用于跨域学习

    Approximation by non-symmetric networks for cross-domain learning. (arXiv:2305.03890v1 [cs.LG])

    [http://arxiv.org/abs/2305.03890](http://arxiv.org/abs/2305.03890)

    本文研究使用非对称内核进行基于内核网络逼近的通用方法，结果表明它可以在跨域学习中显著提高基于内核网络的逼近能力。

    

    在过去的30年中，机器学习在众多过程（如：浅层或深度神经网络逼近、径向基函数网络和各种内核方法）的逼近能力（表达能力）研究中促进了大量的研究。本文针对不变学习、传递学习和合成孔径雷达成像等应用，引入了一种使用非对称内核来研究基于内核网络逼近能力的通用方法。我们考虑使用一组内核的更一般方法，如广义平移网络（其中包括神经网络和平移不变核作为特殊情况）和旋转区函数核。与传统的基于内核的逼近方法不同，我们不能要求内核是正定的。研究结果表明，使用非对称内核可以显著提高内核网络的逼近能力，特别是对于源域和目标域可能在分布上不同的跨域学习。

    For the past 30 years or so, machine learning has stimulated a great deal of research in the study of approximation capabilities (expressive power) of a multitude of processes, such as approximation by shallow or deep neural networks, radial basis function networks, and a variety of kernel based methods. Motivated by applications such as invariant learning, transfer learning, and synthetic aperture radar imaging, we initiate in this paper a general approach to study the approximation capabilities of kernel based networks using non-symmetric kernels. While singular value decomposition is a natural instinct to study such kernels, we consider a more general approach to include the use of a family of kernels, such as generalized translation networks (which include neural networks and translation invariant kernels as special cases) and rotated zonal function kernels. Naturally, unlike traditional kernel based approximation, we cannot require the kernels to be positive definite. Our results 
    

