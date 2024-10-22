# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Capacity: A Measure of the Effective Dimensionality of a Model.](http://arxiv.org/abs/2305.17332) | 学习能力是一种度量模型有效维度的方法，它可以帮助我们判断是否需要获取更多数据或者寻找新的体系结构以提高性能。 |
| [^2] | [A physics-informed neural network framework for modeling obstacle-related equations.](http://arxiv.org/abs/2304.03552) | 本文拓展了基于物理知识的神经网络(PINN) 来解决求解障碍物相关的偏微分方程问题，这种类型的问题需要解决数值方法的难度较大，但作者通过对多种情况的研究证明了PINN的有效性。 |

# 详细

[^1]: 学习能力：模型有效维度的度量方式

    Learning Capacity: A Measure of the Effective Dimensionality of a Model. (arXiv:2305.17332v1 [cs.LG])

    [http://arxiv.org/abs/2305.17332](http://arxiv.org/abs/2305.17332)

    学习能力是一种度量模型有效维度的方法，它可以帮助我们判断是否需要获取更多数据或者寻找新的体系结构以提高性能。

    

    我们利用热力学和推理之间的正式对应关系，将样本数量视为反温度，定义了一种“学习能力”，这是模型有效维度的度量方式。我们发现，对于许多在典型数据集上训练的深度网络，学习能力仅占参数数量的一小部分，取决于用于训练的样本数量，并且在数值上与从PAC-Bayesian框架获得的能力概念一致。学习能力作为测试误差的函数不会出现双峰下降。我们展示了模型的学习能力在非常小和非常大的样本大小处饱和，这提供了指导，说明是否应该获取更多数据或者寻找新的体系结构以提高性能。我们展示了如何使用学习能力来理解有效维数，即使是非参数模型，如随机森林。

    We exploit a formal correspondence between thermodynamics and inference, where the number of samples can be thought of as the inverse temperature, to define a "learning capacity'' which is a measure of the effective dimensionality of a model. We show that the learning capacity is a tiny fraction of the number of parameters for many deep networks trained on typical datasets, depends upon the number of samples used for training, and is numerically consistent with notions of capacity obtained from the PAC-Bayesian framework. The test error as a function of the learning capacity does not exhibit double descent. We show that the learning capacity of a model saturates at very small and very large sample sizes; this provides guidelines, as to whether one should procure more data or whether one should search for new architectures, to improve performance. We show how the learning capacity can be used to understand the effective dimensionality, even for non-parametric models such as random fores
    
[^2]: 基于物理知识的神经网络模型求解障碍物相关方程

    A physics-informed neural network framework for modeling obstacle-related equations. (arXiv:2304.03552v1 [cs.LG])

    [http://arxiv.org/abs/2304.03552](http://arxiv.org/abs/2304.03552)

    本文拓展了基于物理知识的神经网络(PINN) 来解决求解障碍物相关的偏微分方程问题，这种类型的问题需要解决数值方法的难度较大，但作者通过对多种情况的研究证明了PINN的有效性。

    

    深度学习在一些应用中取得了很大成功，但将其用于求解偏微分方程(PDE)　的研究则是近年来的热点，尤其在目前的机器学习库（如TensorFlow或PyTorch）的支持下取得了重大进展。基于物理知识的神经网络（PINN）可通过解析稀疏且噪声数据来求解偏微分方程，是一种有吸引力的工具。本文将拓展PINN来求解障碍物相关PDE，这类方程难度较大，需要可以得到准确解的数值方法。作者在正常和不规则的障碍情况下，对线性和非线性PDE的多个场景进行了演示，证明了所提出的PINNs性能的有效性。

    Deep learning has been highly successful in some applications. Nevertheless, its use for solving partial differential equations (PDEs) has only been of recent interest with current state-of-the-art machine learning libraries, e.g., TensorFlow or PyTorch. Physics-informed neural networks (PINNs) are an attractive tool for solving partial differential equations based on sparse and noisy data. Here extend PINNs to solve obstacle-related PDEs which present a great computational challenge because they necessitate numerical methods that can yield an accurate approximation of the solution that lies above a given obstacle. The performance of the proposed PINNs is demonstrated in multiple scenarios for linear and nonlinear PDEs subject to regular and irregular obstacles.
    

