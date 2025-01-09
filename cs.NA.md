# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Manifold Filter-Combine Networks.](http://arxiv.org/abs/2307.04056) | 这篇论文介绍了一类称为流形滤波-组合网络的大型流形神经网络。作者提出了一种基于构建数据驱动图的方法来实现这种网络，并提供了收敛到连续极限的充分条件，其收敛速度不依赖于滤波器数量。 |

# 详细

[^1]: 流形滤波-组合网络

    Manifold Filter-Combine Networks. (arXiv:2307.04056v1 [stat.ML])

    [http://arxiv.org/abs/2307.04056](http://arxiv.org/abs/2307.04056)

    这篇论文介绍了一类称为流形滤波-组合网络的大型流形神经网络。作者提出了一种基于构建数据驱动图的方法来实现这种网络，并提供了收敛到连续极限的充分条件，其收敛速度不依赖于滤波器数量。

    

    我们介绍了一类大型流形神经网络(MNNs)，我们称之为流形滤波-组合网络。这个类别包括了Wang、Ruiz和Ribeiro之前的研究中考虑的MNNs，流形散射变换(一种基于小波的神经网络模型)，以及其他有趣的之前在文献中未考虑的示例，如Kipf和Welling的图卷积网络的流形等效。然后，我们考虑了一种基于构建数据驱动图的方法，用于在没有对流形有全局知识的情况下实现这样的网络，而只能访问有限数量的样本点。我们提供了网络在样本点数趋于无穷大时能够保证收敛到其连续极限的充分条件。与之前的工作(主要关注特定的MNN结构和图构建)不同，我们的收敛速度并不依赖于使用的滤波器数量。而且，它表现出线性的收敛速度。

    We introduce a large class of manifold neural networks (MNNs) which we call Manifold Filter-Combine Networks. This class includes as special cases, the MNNs considered in previous work by Wang, Ruiz, and Ribeiro, the manifold scattering transform (a wavelet-based model of neural networks), and other interesting examples not previously considered in the literature such as the manifold equivalent of Kipf and Welling's graph convolutional network. We then consider a method, based on building a data-driven graph, for implementing such networks when one does not have global knowledge of the manifold, but merely has access to finitely many sample points. We provide sufficient conditions for the network to provably converge to its continuum limit as the number of sample points tends to infinity. Unlike previous work (which focused on specific MNN architectures and graph constructions), our rate of convergence does not explicitly depend on the number of filters used. Moreover, it exhibits line
    

