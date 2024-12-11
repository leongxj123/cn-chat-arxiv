# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approximating Langevin Monte Carlo with ResNet-like Neural Network architectures.](http://arxiv.org/abs/2311.03242) | 本论文提出了一种使用类似ResNet的神经网络架构来近似Langevin Monte Carlo算法，通过将来自简单参考分布的样本映射到目标分布的样本中来进行采样，具有较好的逼近速度和表达性。 |

# 详细

[^1]: 使用类似ResNet的神经网络架构近似Langevin Monte Carlo

    Approximating Langevin Monte Carlo with ResNet-like Neural Network architectures. (arXiv:2311.03242v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.03242](http://arxiv.org/abs/2311.03242)

    本论文提出了一种使用类似ResNet的神经网络架构来近似Langevin Monte Carlo算法，通过将来自简单参考分布的样本映射到目标分布的样本中来进行采样，具有较好的逼近速度和表达性。

    

    我们通过构建一个神经网络，将来自简单参考分布（如标准正态分布）的样本映射到目标分布的样本中，从而从给定的目标分布中进行采样。为此，我们提出使用受Langevin Monte Carlo (LMC)算法启发的神经网络架构。基于LMC扰动结果，在Wasserstein-2距离上，我们展示了该架构对于平滑的对数凹目标分布的逼近速度。分析严重依赖于扰动LMC过程的中间度量的亚高斯性概念。特别地，我们根据不同扰动假设推导出了中间方差代理的增长界限。此外，我们提出了一种类似于深度残差神经网络的架构，并推导出了近似样本与目标分布映射的表达性结果。

    We sample from a given target distribution by constructing a neural network which maps samples from a simple reference, e.g. the standard normal distribution, to samples from the target. To that end, we propose using a neural network architecture inspired by the Langevin Monte Carlo (LMC) algorithm. Based on LMC perturbation results, we show approximation rates of the proposed architecture for smooth, log-concave target distributions measured in the Wasserstein-$2$ distance. The analysis heavily relies on the notion of sub-Gaussianity of the intermediate measures of the perturbed LMC process. In particular, we derive bounds on the growth of the intermediate variance proxies under different assumptions on the perturbations. Moreover, we propose an architecture similar to deep residual neural networks and derive expressivity results for approximating the sample to target distribution map.
    

