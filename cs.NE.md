# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Universal Approximation Theorem for Vector- and Hypercomplex-Valued Neural Networks.](http://arxiv.org/abs/2401.02277) | 该论文通过引入非退化代数的概念，扩展了通用逼近定理，使其适用于广泛的向量值神经网络，包括超复值模型。这对于神经网络在回归和分类任务等多种应用中的应用具有重要意义。 |
| [^2] | [Active Learning in Symbolic Regression Performance with Physical Constraints.](http://arxiv.org/abs/2305.10379) | 本文探讨了利用进化符号回归作为主动学习中的方法来提出哪些数据应该被采集，通过“委员会查询”来减少所需数据，并在重新发现已知方程所需的数据方面实现最新的结果。 |

# 详细

[^1]: 向量值和超复值神经网络的通用逼近定理

    Universal Approximation Theorem for Vector- and Hypercomplex-Valued Neural Networks. (arXiv:2401.02277v1 [cs.LG])

    [http://arxiv.org/abs/2401.02277](http://arxiv.org/abs/2401.02277)

    该论文通过引入非退化代数的概念，扩展了通用逼近定理，使其适用于广泛的向量值神经网络，包括超复值模型。这对于神经网络在回归和分类任务等多种应用中的应用具有重要意义。

    

    通用逼近定理表明，具有一层隐藏层的神经网络可以以任意所需的精度逼近紧集上的连续函数。该定理支持了神经网络在回归和分类任务等各种应用中的使用。此外，对于实值神经网络和一些超复值神经网络（例如复数、四元数、四元数矢量和Clifford值神经网络），该定理均有效。然而，超复值神经网络是一种在具有附加代数或几何性质的代数上定义的向量值神经网络。本文将通用逼近定理扩展到了广泛的向量值神经网络，包括超复值模型作为特殊实例。具体而言，我们引入了非退化代数的概念，并阐述了在这种代数上定义的神经网络的通用逼近定理。

    The universal approximation theorem states that a neural network with one hidden layer can approximate continuous functions on compact sets with any desired precision. This theorem supports using neural networks for various applications, including regression and classification tasks. Furthermore, it is valid for real-valued neural networks and some hypercomplex-valued neural networks such as complex-, quaternion-, tessarine-, and Clifford-valued neural networks. However, hypercomplex-valued neural networks are a type of vector-valued neural network defined on an algebra with additional algebraic or geometric properties. This paper extends the universal approximation theorem for a wide range of vector-valued neural networks, including hypercomplex-valued models as particular instances. Precisely, we introduce the concept of non-degenerate algebra and state the universal approximation theorem for neural networks defined on such algebras.
    
[^2]: 基于物理约束的符号回归中主动学习的表现

    Active Learning in Symbolic Regression Performance with Physical Constraints. (arXiv:2305.10379v1 [cs.LG])

    [http://arxiv.org/abs/2305.10379](http://arxiv.org/abs/2305.10379)

    本文探讨了利用进化符号回归作为主动学习中的方法来提出哪些数据应该被采集，通过“委员会查询”来减少所需数据，并在重新发现已知方程所需的数据方面实现最新的结果。

    

    进化符号回归（SR）是一种将符号方程拟合到数据中的方法，可以得到简洁易懂的模型。本文探讨使用SR作为主动学习中的方法来提出哪些数据应该被采集，在此过程中考虑物理约束。基于主动学习的SR通过“委员会查询”来提出下一步实验。物理约束可以在非常低的数据情况下改善所建议的方程。这些方法可以减少SR所需的数据，并在重新发现已知方程所需的数据方面实现最新的结果。

    Evolutionary symbolic regression (SR) fits a symbolic equation to data, which gives a concise interpretable model. We explore using SR as a method to propose which data to gather in an active learning setting with physical constraints. SR with active learning proposes which experiments to do next. Active learning is done with query by committee, where the Pareto frontier of equations is the committee. The physical constraints improve proposed equations in very low data settings. These approaches reduce the data required for SR and achieves state of the art results in data required to rediscover known equations.
    

