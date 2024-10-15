# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Two-scale Neural Networks for Partial Differential Equations with Small Parameters](https://arxiv.org/abs/2402.17232) | 提出了一种用双尺度神经网络方法解决具有小参数的偏微分方程的方法，能够直接将小参数纳入神经网络架构中，从而简化解决过程，并能够合理准确地捕捉由小参数引起的解中大导数特征。 |
| [^2] | [Weighted variation spaces and approximation by shallow ReLU networks.](http://arxiv.org/abs/2307.15772) | 本文研究了在有界域上通过单隐藏层ReLU网络逼近函数的问题，介绍了新的模型类定义加权变差空间，该定义与域本身相关。 |
| [^3] | [Rank-adaptive spectral pruning of convolutional layers during training.](http://arxiv.org/abs/2305.19059) | 本论文提出了一种新的低参数训练方法，该方法将卷积分解为张量Tucker格式，并在训练过程中自适应地修剪卷积核的Tucker秩，可以有效地降低训练成本。 |

# 详细

[^1]: 具有小参数的偏微分方程的双尺度神经网络

    Two-scale Neural Networks for Partial Differential Equations with Small Parameters

    [https://arxiv.org/abs/2402.17232](https://arxiv.org/abs/2402.17232)

    提出了一种用双尺度神经网络方法解决具有小参数的偏微分方程的方法，能够直接将小参数纳入神经网络架构中，从而简化解决过程，并能够合理准确地捕捉由小参数引起的解中大导数特征。

    

    我们提出了一种用物理信息神经网络（PINNs）解决具有小参数的偏微分方程（PDEs）的双尺度神经网络方法。我们直接将小参数纳入神经网络的架构中。所提出的方法使得以简单方式解决具有小参数的PDE成为可能，而无需添加傅里叶特征或其他计算繁琐的截断参数搜索。多个数值例子展示了在解决由小参数引起的解中大导数特征时的合理准确性。

    arXiv:2402.17232v1 Announce Type: cross  Abstract: We propose a two-scale neural network method for solving partial differential equations (PDEs) with small parameters using physics-informed neural networks (PINNs). We directly incorporate the small parameters into the architecture of neural networks. The proposed method enables solving PDEs with small parameters in a simple fashion, without adding Fourier features or other computationally taxing searches of truncation parameters. Various numerical examples demonstrate reasonable accuracy in capturing features of large derivatives in the solutions caused by small parameters.
    
[^2]: 加权变差空间与浅层ReLU网络的逼近

    Weighted variation spaces and approximation by shallow ReLU networks. (arXiv:2307.15772v1 [stat.ML])

    [http://arxiv.org/abs/2307.15772](http://arxiv.org/abs/2307.15772)

    本文研究了在有界域上通过单隐藏层ReLU网络逼近函数的问题，介绍了新的模型类定义加权变差空间，该定义与域本身相关。

    

    本文研究了在有界域Ω⊂Rd上，通过宽度为n的单隐藏层ReLU神经网络的输出来逼近函数f的情况。这种非线性的n项字典逼近已经得到广泛研究，因为它是神经网络逼近(NNA)的最简单情况。对于这种NNA形式，有几个著名的逼近结果，引入了在Ω上的函数的新型模型类，其逼近速率避免了维数灾难。这些新型模型类包括Barron类和基于稀疏性或变差的类，例如Radon域BV类。本文关注于在域Ω上定义这些新型模型类。当前这些模型类的定义不依赖于域Ω。通过引入加权变差空间的概念，给出了关于域的更恰当的模型类定义。这些新型模型类与域本身相关。

    We investigate the approximation of functions $f$ on a bounded domain $\Omega\subset \mathbb{R}^d$ by the outputs of single-hidden-layer ReLU neural networks of width $n$. This form of nonlinear $n$-term dictionary approximation has been intensely studied since it is the simplest case of neural network approximation (NNA). There are several celebrated approximation results for this form of NNA that introduce novel model classes of functions on $\Omega$ whose approximation rates avoid the curse of dimensionality. These novel classes include Barron classes, and classes based on sparsity or variation such as the Radon-domain BV classes.  The present paper is concerned with the definition of these novel model classes on domains $\Omega$. The current definition of these model classes does not depend on the domain $\Omega$. A new and more proper definition of model classes on domains is given by introducing the concept of weighted variation spaces. These new model classes are intrinsic to th
    
[^3]: 训练期间的自适应秩谱剪枝卷积层

    Rank-adaptive spectral pruning of convolutional layers during training. (arXiv:2305.19059v1 [cs.LG])

    [http://arxiv.org/abs/2305.19059](http://arxiv.org/abs/2305.19059)

    本论文提出了一种新的低参数训练方法，该方法将卷积分解为张量Tucker格式，并在训练过程中自适应地修剪卷积核的Tucker秩，可以有效地降低训练成本。

    

    深度学习模型在计算成本和内存需求方面增长迅速，因此已经发展了各种剪枝技术以减少模型参数。大多数技术侧重于通过在完整训练后对网络进行修剪以减少推理成本。少量的方法解决了减少训练成本的问题，主要是通过低秩层分解来压缩网络。尽管这些方法对于线性层是有效的，但是它们无法有效处理卷积滤波器。在这项工作中，我们提出了一种低参数训练方法，将卷积分解为张量Tucker格式，并在训练过程中自适应地修剪卷积核的Tucker秩。利用微分方程在张量流形上的几何积分理论的基本结果，我们获得了一个鲁棒的训练算法，证明能够逼近完整的基线性能并保证损失下降。

    The computing cost and memory demand of deep learning pipelines have grown fast in recent years and thus a variety of pruning techniques have been developed to reduce model parameters. The majority of these techniques focus on reducing inference costs by pruning the network after a pass of full training. A smaller number of methods address the reduction of training costs, mostly based on compressing the network via low-rank layer factorizations. Despite their efficiency for linear layers, these methods fail to effectively handle convolutional filters. In this work, we propose a low-parametric training method that factorizes the convolutions into tensor Tucker format and adaptively prunes the Tucker ranks of the convolutional kernel during training. Leveraging fundamental results from geometric integration theory of differential equations on tensor manifolds, we obtain a robust training algorithm that provably approximates the full baseline performance and guarantees loss descent. A var
    

