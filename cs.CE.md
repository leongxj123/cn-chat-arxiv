# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Training dynamics in Physics-Informed Neural Networks with feature mapping](https://arxiv.org/abs/2402.06955) | 本研究探究了使用特征映射层的物理引导神经网络（PINNs）的训练动态，通过极限共轭核和神经切向核揭示了模型的收敛和泛化。我们提出了一种替代基于傅里叶变换的特征映射的条件正定径向基函数，证明了该方法在各种问题集中的有效性，并可以轻松实现在坐标输入网络中。这为广泛的PINNs研究带来了益处。 |

# 详细

[^1]: 使用特征映射的物理引导神经网络中的训练动态

    Training dynamics in Physics-Informed Neural Networks with feature mapping

    [https://arxiv.org/abs/2402.06955](https://arxiv.org/abs/2402.06955)

    本研究探究了使用特征映射层的物理引导神经网络（PINNs）的训练动态，通过极限共轭核和神经切向核揭示了模型的收敛和泛化。我们提出了一种替代基于傅里叶变换的特征映射的条件正定径向基函数，证明了该方法在各种问题集中的有效性，并可以轻松实现在坐标输入网络中。这为广泛的PINNs研究带来了益处。

    

    物理引导神经网络（PINNs）已成为解决偏微分方程（PDE）的标志性机器学习方法。尽管其变体取得了显著进展，但来自更广泛的隐式神经表示研究的特征映射的经验性成功在很大程度上被忽视。我们通过极限共轭核和神经切向核来研究带有特征映射层的PINNs的训练动态，从而揭示了模型的收敛和泛化。我们还展示了常用的基于傅里叶变换的特征映射在某些情况下的不足，并提出条件正定的径向基函数作为更好的替代方法。经验证实，我们的方法在各种正向和反向问题集中非常有效。这种简单的技术可以轻松在坐标输入网络中实现，并受益于广泛的PINNs研究。

    Physics-Informed Neural Networks (PINNs) have emerged as an iconic machine learning approach for solving Partial Differential Equations (PDEs). Although its variants have achieved significant progress, the empirical success of utilising feature mapping from the wider Implicit Neural Representations studies has been substantially neglected. We investigate the training dynamics of PINNs with a feature mapping layer via the limiting Conjugate Kernel and Neural Tangent Kernel, which sheds light on the convergence and generalisation of the model. We also show the inadequacy of commonly used Fourier-based feature mapping in some scenarios and propose the conditional positive definite Radial Basis Function as a better alternative. The empirical results reveal the efficacy of our method in diverse forward and inverse problem sets. This simple technique can be easily implemented in coordinate input networks and benefits the broad PINNs research.
    

