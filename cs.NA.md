# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Functional SDE approximation inspired by a deep operator network architecture](https://arxiv.org/abs/2402.03028) | 本文提出了一种受深度算子网络结构启发的函数SDE近似方法，通过深度神经网络和多项式混沌展开实现对随机微分方程解的近似，并通过学习减轻指数级复杂度的问题。 |

# 详细

[^1]: 受深度算子网络结构启发的函数SDE近似方法

    Functional SDE approximation inspired by a deep operator network architecture

    [https://arxiv.org/abs/2402.03028](https://arxiv.org/abs/2402.03028)

    本文提出了一种受深度算子网络结构启发的函数SDE近似方法，通过深度神经网络和多项式混沌展开实现对随机微分方程解的近似，并通过学习减轻指数级复杂度的问题。

    

    本文提出并分析了一种通过深度神经网络近似随机微分方程（SDE）解的新方法。该结构灵感来自于深度算子网络（DeepONets）的概念，它基于函数空间中的算子学习，以及在网络中表示的降维基础。在我们的设置中，我们利用了随机过程的多项式混沌展开（PCE），并将相应的架构称为SDEONet。在参数化偏微分方程的不确定性量化（UQ）领域中，PCE被广泛使用。然而，在SDE中并非如此，传统的采样方法占主导地位，而功能性方法很少见。截断的PCE存在一个主要挑战，即随着最大多项式阶数和基函数数量的增加，分量的数量呈指数级增长。所提出的SDEONet结构旨在通过学习来减轻指数级复杂度的问题。

    A novel approach to approximate solutions of Stochastic Differential Equations (SDEs) by Deep Neural Networks is derived and analysed. The architecture is inspired by the notion of Deep Operator Networks (DeepONets), which is based on operator learning in function spaces in terms of a reduced basis also represented in the network. In our setting, we make use of a polynomial chaos expansion (PCE) of stochastic processes and call the corresponding architecture SDEONet. The PCE has been used extensively in the area of uncertainty quantification (UQ) with parametric partial differential equations. This however is not the case with SDE, where classical sampling methods dominate and functional approaches are seen rarely. A main challenge with truncated PCEs occurs due to the drastic growth of the number of components with respect to the maximum polynomial degree and the number of basis elements. The proposed SDEONet architecture aims to alleviate the issue of exponential complexity by learni
    

