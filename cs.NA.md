# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Good Lattice Training: Physics-Informed Neural Networks Accelerated by Number Theory.](http://arxiv.org/abs/2307.13869) | 本研究提出了一种新的物理信息神经网络训练方法，受数论方法启发，通过选择适当的插值点来提高解决偏微分方程的准确性和效率。 |
| [^2] | [Mathematical analysis of singularities in the diffusion model under the submanifold assumption.](http://arxiv.org/abs/2301.07882) | 本文提供了扩散模型中漂移项的数学分析。通过次流形假设，提出一种新的目标函数和相关的损失函数，可处理低维流形上的奇异数据分布，解决了均值漂移函数和得分函数渐近发散的问题。 |

# 详细

[^1]: 优良格训练: 借助数论加速的物理信息神经网络

    Good Lattice Training: Physics-Informed Neural Networks Accelerated by Number Theory. (arXiv:2307.13869v1 [cs.LG])

    [http://arxiv.org/abs/2307.13869](http://arxiv.org/abs/2307.13869)

    本研究提出了一种新的物理信息神经网络训练方法，受数论方法启发，通过选择适当的插值点来提高解决偏微分方程的准确性和效率。

    

    物理信息神经网络(PINNs)提供了一种新颖高效的解决偏微分方程(PDEs)的方法。它们的成功在于物理信息损失函数，该函数训练神经网络以满足给定点上的PDE，并对解进行逼近。然而，PDE的解在本质上是无限维的，并且输出与解之间的距离是定义在整个域上的积分。因此，物理信息损失函数仅提供有限的逼近。在选择合适的插值点方面则变得至关重要，尽管这一方面经常被忽视。在本文中，我们提出了一种新的技术，称为优良格训练(GLT)，用于PINNs，受数值分析中的数论方法的启发。GLT提供了一组即使在少量点和多维空间中也非常有效的插值点。我们的实验表明，GLT只需要2-20倍的点数

    Physics-informed neural networks (PINNs) offer a novel and efficient approach to solving partial differential equations (PDEs). Their success lies in the physics-informed loss, which trains a neural network to satisfy a given PDE at specific points and to approximate the solution. However, the solutions to PDEs are inherently infinite-dimensional, and the distance between the output and the solution is defined by an integral over the domain. Therefore, the physics-informed loss only provides a finite approximation, and selecting appropriate collocation points becomes crucial to suppress the discretization errors, although this aspect has often been overlooked. In this paper, we propose a new technique called good lattice training (GLT) for PINNs, inspired by number theoretic methods for numerical analysis. GLT offers a set of collocation points that are effective even with a small number of points and for multi-dimensional spaces. Our experiments demonstrate that GLT requires 2--20 tim
    
[^2]: 基于次流形假设下扩散模型奇异性的数学分析

    Mathematical analysis of singularities in the diffusion model under the submanifold assumption. (arXiv:2301.07882v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.07882](http://arxiv.org/abs/2301.07882)

    本文提供了扩散模型中漂移项的数学分析。通过次流形假设，提出一种新的目标函数和相关的损失函数，可处理低维流形上的奇异数据分布，解决了均值漂移函数和得分函数渐近发散的问题。

    

    本文提供了机器学习中扩散模型的数学分析。以条件期望表示反向采样流程的漂移项，其中涉及数据分布和前向扩散。训练过程旨在通过最小化与条件期望相关的均方残差来寻找此类漂移函数。使用前向扩散的Green函数的小时间近似，我们证明了DDPM中的解析均值漂移函数和SGM中的得分函数在采样过程的最后阶段，对于像那些集中在低维流形上的奇异数据分布而言，渐近地发散，因此难以通过网络进行逼近。为了克服这个困难，我们推导出了一个新的目标函数和相关的损失函数，即使在处理奇异数据分布时仍然保持有界。我们通过几个数值实验来说明理论发现。

    This paper provide several mathematical analyses of the diffusion model in machine learning. The drift term of the backwards sampling process is represented as a conditional expectation involving the data distribution and the forward diffusion. The training process aims to find such a drift function by minimizing the mean-squared residue related to the conditional expectation. Using small-time approximations of the Green's function of the forward diffusion, we show that the analytical mean drift function in DDPM and the score function in SGM asymptotically blow up in the final stages of the sampling process for singular data distributions such as those concentrated on lower-dimensional manifolds, and is therefore difficult to approximate by a network. To overcome this difficulty, we derive a new target function and associated loss, which remains bounded even for singular data distributions. We illustrate the theoretical findings with several numerical examples.
    

