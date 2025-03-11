# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The curse of dimensionality in operator learning.](http://arxiv.org/abs/2306.15924) | 算子学习中存在维度诅咒，但对于由Hamilton-Jacobi方程定义的解算子可以克服维度诅咒。 |

# 详细

[^1]: 运算学习中的维度诅咒

    The curse of dimensionality in operator learning. (arXiv:2306.15924v1 [cs.LG])

    [http://arxiv.org/abs/2306.15924](http://arxiv.org/abs/2306.15924)

    算子学习中存在维度诅咒，但对于由Hamilton-Jacobi方程定义的解算子可以克服维度诅咒。

    

    神经算子架构利用神经网络来近似映射函数空间之间的算子，可以用于通过模拟加速模型评估，或者从数据中发现模型。因此，这一方法在近年来受到越来越多的关注，引发了算子学习领域的快速发展。本文的第一项贡献是证明了对于一般的只由其 $C^r$ 或 Lipschitz 正则性特征化的算子类，算子学习遭受了维度诅咒，这里通过无穷维输入和输出函数空间的表征来精确定义维度诅咒。该结果适用于包括 PCA-Net、DeepONet 和 FNO 在内的多种现有神经算子。本文的第二项贡献是证明了对于由Hamilton-Jacobi方程定义的解算子，可以克服一般的维度诅咒；这是通过引入新的表示方法来实现的。

    Neural operator architectures employ neural networks to approximate operators mapping between Banach spaces of functions; they may be used to accelerate model evaluations via emulation, or to discover models from data. Consequently, the methodology has received increasing attention over recent years, giving rise to the rapidly growing field of operator learning. The first contribution of this paper is to prove that for general classes of operators which are characterized only by their $C^r$- or Lipschitz-regularity, operator learning suffers from a curse of dimensionality, defined precisely here in terms of representations of the infinite-dimensional input and output function spaces. The result is applicable to a wide variety of existing neural operators, including PCA-Net, DeepONet and the FNO. The second contribution of the paper is to prove that the general curse of dimensionality can be overcome for solution operators defined by the Hamilton-Jacobi equation; this is achieved by lev
    

