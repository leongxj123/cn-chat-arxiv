# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Low-Order Discontinuous Galerkin Methods with Neural Ordinary Differential Equations for Compressible Navier--Stokes Equations.](http://arxiv.org/abs/2310.18897) | 本研究提出了一种方法，通过在不连续Galerkin方法中加入神经常微分方程，学习子网格尺度模型的效果，从而提高模拟的准确性和加速计算过程。 |

# 详细

[^1]: 使用神经常微分方程增强低阶不连续Galerkin方法在可压Navier-Stokes方程中的应用

    Enhancing Low-Order Discontinuous Galerkin Methods with Neural Ordinary Differential Equations for Compressible Navier--Stokes Equations. (arXiv:2310.18897v2 [physics.flu-dyn] UPDATED)

    [http://arxiv.org/abs/2310.18897](http://arxiv.org/abs/2310.18897)

    本研究提出了一种方法，通过在不连续Galerkin方法中加入神经常微分方程，学习子网格尺度模型的效果，从而提高模拟的准确性和加速计算过程。

    

    随着计算能力的增长，模拟变得更加复杂和准确。然而，高保真度的模拟需要巨大的计算资源。为了降低计算成本，通常会运行一个低保真度模型并采用子网格尺度模型，但选择适当的子网格尺度模型并对其进行调节是具有挑战性的。我们在不连续Galerkin（DG）空间离散化的背景下提出了一种新颖的方法，通过在偏微分方程模拟中引入神经常微分算子来学习子网格尺度模型的效果。我们的方法在连续级别上学习低阶DG求解器中缺失的尺度，从而提高低阶DG近似的准确性，同时以一定程度的精度加速滤波高阶DG模拟。我们通过实验证明了我们方法的性能。

    The growing computing power over the years has enabled simulations to become more complex and accurate. While immensely valuable for scientific discovery and problem-solving, however, high-fidelity simulations come with significant computational demands. As a result, it is common to run a low-fidelity model with a subgrid-scale model to reduce the computational cost, but selecting the appropriate subgrid-scale models and tuning them are challenging. We propose a novel method for learning the subgrid-scale model effects when simulating partial differential equations augmented by neural ordinary differential operators in the context of discontinuous Galerkin (DG) spatial discretization. Our approach learns the missing scales of the low-order DG solver at a continuous level and hence improves the accuracy of the low-order DG approximations as well as accelerates the filtered high-order DG simulations with a certain degree of precision. We demonstrate the performance of our approach throug
    

