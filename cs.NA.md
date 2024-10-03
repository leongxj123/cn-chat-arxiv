# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HAMLET: Graph Transformer Neural Operator for Partial Differential Equations](https://arxiv.org/abs/2402.03541) | HAMLET是一个图变换神经算子框架，通过使用模块化输入编码器将微分方程信息直接融入解决过程中，并展示出在处理复杂数据和噪声方面的鲁棒性，适用于任意几何形状和各种输入格式的PDE问题。通过大量实验，我们证明了HAMLET能够超越当前的PDE技术。 |
| [^2] | [Efficient error and variance estimation for randomized matrix computations](https://arxiv.org/abs/2207.06342) | 该论文提出了用于随机矩阵计算的高效误差和方差估计方法，可帮助评估输出质量并指导算法参数选择。 |

# 详细

[^1]: HAMLET：用于偏微分方程的图变换神经算子

    HAMLET: Graph Transformer Neural Operator for Partial Differential Equations

    [https://arxiv.org/abs/2402.03541](https://arxiv.org/abs/2402.03541)

    HAMLET是一个图变换神经算子框架，通过使用模块化输入编码器将微分方程信息直接融入解决过程中，并展示出在处理复杂数据和噪声方面的鲁棒性，适用于任意几何形状和各种输入格式的PDE问题。通过大量实验，我们证明了HAMLET能够超越当前的PDE技术。

    

    我们提出了一种新颖的图变换框架HAMLET，旨在解决使用神经网络求解偏微分方程（PDE）时的挑战。该框架使用具有模块化输入编码器的图变换器，将微分方程信息直接融入解决过程中。这种模块化增强了参数对应控制，使得HAMLET能够适应任意几何形状和各种输入格式的PDE。值得注意的是，HAMLET能够有效扩展到处理复杂数据和噪声，展示出其鲁棒性。HAMLET不仅适用于单一类型的物理模拟，还可以应用于各个领域。此外，它提升了模型的弹性和性能，特别是在数据有限的场景下。通过大量实验，我们证明了我们的框架能够超越当前的PDE技术。

    We present a novel graph transformer framework, HAMLET, designed to address the challenges in solving partial differential equations (PDEs) using neural networks. The framework uses graph transformers with modular input encoders to directly incorporate differential equation information into the solution process. This modularity enhances parameter correspondence control, making HAMLET adaptable to PDEs of arbitrary geometries and varied input formats. Notably, HAMLET scales effectively with increasing data complexity and noise, showcasing its robustness. HAMLET is not just tailored to a single type of physical simulation, but can be applied across various domains. Moreover, it boosts model resilience and performance, especially in scenarios with limited data. We demonstrate, through extensive experiments, that our framework is capable of outperforming current techniques for PDEs.
    
[^2]: 针对随机矩阵计算的高效误差和方差估计

    Efficient error and variance estimation for randomized matrix computations

    [https://arxiv.org/abs/2207.06342](https://arxiv.org/abs/2207.06342)

    该论文提出了用于随机矩阵计算的高效误差和方差估计方法，可帮助评估输出质量并指导算法参数选择。

    

    随机矩阵算法已成为科学计算和机器学习中必不可少的工具。为了安全地在应用中使用这些算法，需要结合后验误差估计来评估输出的质量。为满足这一需求，本文提出了两种诊断方法：用于随机低秩逼近的留一法误差估计器和一种杰基刀重采样方法，用于估计随机矩阵计算的输出方差。这两种诊断方法对于随机低秩逼近算法（如随机奇异值分解和随机Nystrom逼近）计算迅速，并提供可用于评估计算输出质量和指导算法参数选择的有用信息。

    arXiv:2207.06342v4 Announce Type: replace-cross  Abstract: Randomized matrix algorithms have become workhorse tools in scientific computing and machine learning. To use these algorithms safely in applications, they should be coupled with posterior error estimates to assess the quality of the output. To meet this need, this paper proposes two diagnostics: a leave-one-out error estimator for randomized low-rank approximations and a jackknife resampling method to estimate the variance of the output of a randomized matrix computation. Both of these diagnostics are rapid to compute for randomized low-rank approximation algorithms such as the randomized SVD and randomized Nystr\"om approximation, and they provide useful information that can be used to assess the quality of the computed output and guide algorithmic parameter choices.
    

