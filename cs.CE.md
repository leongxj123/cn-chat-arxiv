# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving the Adaptive Moment Estimation (ADAM) stochastic optimizer through an Implicit-Explicit (IMEX) time-stepping approach](https://arxiv.org/abs/2403.13704) | 通过隐式显式(IMEX)时间步进方法改进自适应矩估计（ADAM）随机优化器，提出了一种新的神经网络训练优化算法，比经典Adam在几个回归和分类问题上表现更好。 |

# 详细

[^1]: 通过隐式显式(IMEX)时间步进方法改进自适应矩估计（ADAM）随机优化器

    Improving the Adaptive Moment Estimation (ADAM) stochastic optimizer through an Implicit-Explicit (IMEX) time-stepping approach

    [https://arxiv.org/abs/2403.13704](https://arxiv.org/abs/2403.13704)

    通过隐式显式(IMEX)时间步进方法改进自适应矩估计（ADAM）随机优化器，提出了一种新的神经网络训练优化算法，比经典Adam在几个回归和分类问题上表现更好。

    

    Adam优化器通常用于神经网络训练中，对应于在非常小的学习速率限制下的基本常微分方程（ODE）。本文表明，经典Adam算法是底层ODE的一阶隐式显式(IMEX) Euler离散化。从时间离散化角度出发，我们提出了通过使用更高阶IMEX方法来解决ODE的Adam方案的新扩展。基于这种方法，我们推导了一种新的神经网络训练优化算法，在几个回归和分类问题上比经典Adam表现更好。

    arXiv:2403.13704v1 Announce Type: cross  Abstract: The Adam optimizer, often used in Machine Learning for neural network training, corresponds to an underlying ordinary differential equation (ODE) in the limit of very small learning rates. This work shows that the classical Adam algorithm is a first order implicit-explicit (IMEX) Euler discretization of the underlying ODE. Employing the time discretization point of view, we propose new extensions of the Adam scheme obtained by using higher order IMEX methods to solve the ODE. Based on this approach, we derive a new optimization algorithm for neural network training that performs better than classical Adam on several regression and classification problems.
    

