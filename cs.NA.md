# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discovering Artificial Viscosity Models for Discontinuous Galerkin Approximation of Conservation Laws using Physics-Informed Machine Learning](https://arxiv.org/abs/2402.16517) | 使用物理信息机器学习算法自动发现人工粘性模型，无需数据集训练，成功应用于高阶守恒定律求解器中。 |
| [^2] | [Solving PDEs on Spheres with Physics-Informed Convolutional Neural Networks.](http://arxiv.org/abs/2308.09605) | 本文严格分析了在球面上解决PDEs的物理信息卷积神经网络（PICNN），通过使用最新的逼近结果和球谐分析，证明了逼近误差与Sobolev范数的上界，并建立了快速收敛速率。实验结果也验证了理论分析的有效性。 |

# 详细

[^1]: 使用物理信息机器学习发现不连续Galerkin逼近守恒定律的人工粘性模型

    Discovering Artificial Viscosity Models for Discontinuous Galerkin Approximation of Conservation Laws using Physics-Informed Machine Learning

    [https://arxiv.org/abs/2402.16517](https://arxiv.org/abs/2402.16517)

    使用物理信息机器学习算法自动发现人工粘性模型，无需数据集训练，成功应用于高阶守恒定律求解器中。

    

    基于有限元的高阶守恒定律求解器提供了较高的准确性，但在不连续处面临Gibbs现象挑战。人工粘性是基于物理见解的解决方案。本文提出了一种物理信息机器学习算法，用于自动发现非监督范式下的人工粘性模型。该算法受强化学习启发，通过最小化定义为相对参考解的差异的损失来训练神经网络，单元格逐个单元格操作（粘性模型）。这使得能够进行无数据集的训练过程。我们证明了该算法通过将其整合到最先进的Runge-Kutta不连续Galerkin求解器中是有效的。我们在标量和矢量问题上展示了几个数值测试，如Burgers'和Euler的方程在一维和二维的情况。

    arXiv:2402.16517v1 Announce Type: cross  Abstract: Finite element-based high-order solvers of conservation laws offer large accuracy but face challenges near discontinuities due to the Gibbs phenomenon. Artificial viscosity is a popular and effective solution to this problem based on physical insight. In this work, we present a physics-informed machine learning algorithm to automate the discovery of artificial viscosity models in a non-supervised paradigm. The algorithm is inspired by reinforcement learning and trains a neural network acting cell-by-cell (the viscosity model) by minimizing a loss defined as the difference with respect to a reference solution thanks to automatic differentiation. This enables a dataset-free training procedure. We prove that the algorithm is effective by integrating it into a state-of-the-art Runge-Kutta discontinuous Galerkin solver. We showcase several numerical tests on scalar and vectorial problems, such as Burgers' and Euler's equations in one and tw
    
[^2]: 使用物理信息卷积神经网络在球面上解决偏微分方程

    Solving PDEs on Spheres with Physics-Informed Convolutional Neural Networks. (arXiv:2308.09605v1 [math.NA])

    [http://arxiv.org/abs/2308.09605](http://arxiv.org/abs/2308.09605)

    本文严格分析了在球面上解决PDEs的物理信息卷积神经网络（PICNN），通过使用最新的逼近结果和球谐分析，证明了逼近误差与Sobolev范数的上界，并建立了快速收敛速率。实验结果也验证了理论分析的有效性。

    

    物理信息神经网络（PINNs）已被证明在解决各种实验角度中的偏微分方程（PDEs）方面非常高效。一些最近的研究还提出了针对表面，包括球面上的PDEs的PINN算法。然而，对于PINNs的数值性能，尤其是在表面或流形上的PINNs，仍然缺乏理论理解。本文中，我们对用于在球面上解决PDEs的物理信息卷积神经网络（PICNN）进行了严格分析。通过使用和改进深度卷积神经网络和球谐分析的最新逼近结果，我们证明了该逼近误差与Sobolev范数的上界。随后，我们将这一结果与创新的局部复杂度分析相结合，建立了PICNN的快速收敛速率。我们的理论结果也得到了实验的验证和补充。鉴于这些发现，

    Physics-informed neural networks (PINNs) have been demonstrated to be efficient in solving partial differential equations (PDEs) from a variety of experimental perspectives. Some recent studies have also proposed PINN algorithms for PDEs on surfaces, including spheres. However, theoretical understanding of the numerical performance of PINNs, especially PINNs on surfaces or manifolds, is still lacking. In this paper, we establish rigorous analysis of the physics-informed convolutional neural network (PICNN) for solving PDEs on the sphere. By using and improving the latest approximation results of deep convolutional neural networks and spherical harmonic analysis, we prove an upper bound for the approximation error with respect to the Sobolev norm. Subsequently, we integrate this with innovative localization complexity analysis to establish fast convergence rates for PICNN. Our theoretical results are also confirmed and supplemented by our experiments. In light of these findings, we expl
    

