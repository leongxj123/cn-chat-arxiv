# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors.](http://arxiv.org/abs/2401.13652) | 本文提出了一种利用图信息神经网络和稀疏网格来检测不连续函数不连续界面的新方法，该方法在维度大于3的情况下表现出高效且准确的不连续性检测能力，在维度n = 2和n = 4的函数上进行的实验验证了其高效性和泛化能力，并具有可移植性和多功能性。 |

# 详细

[^1]: 基于稀疏网格的不连续性检测的图信息神经网络

    Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors. (arXiv:2401.13652v1 [cs.LG])

    [http://arxiv.org/abs/2401.13652](http://arxiv.org/abs/2401.13652)

    本文提出了一种利用图信息神经网络和稀疏网格来检测不连续函数不连续界面的新方法，该方法在维度大于3的情况下表现出高效且准确的不连续性检测能力，在维度n = 2和n = 4的函数上进行的实验验证了其高效性和泛化能力，并具有可移植性和多功能性。

    

    本文提出了一种新颖的方法来检测不连续函数的不连续界面。该方法利用了基于图的神经网络（GINNs）和稀疏网格来解决维度大于3的情况下的不连续性检测。训练过的GINNs在稀疏网格上识别有问题的点，并利用构建在网格上的图结构实现高效准确的不连续性检测性能。我们还引入了一种递归算法用于一般的基于稀疏网格的检测器，具有收敛性和易于应用性。在维度n=2和n=4的函数上进行的数值实验证明了GINNs在检测不连续界面方面的高效性和鲁棒泛化能力。值得注意的是，经过训练的GINNs具有可移植性和多功能性，可以集成到各种算法中并共享给用户。

    In this paper, we present a novel approach for detecting the discontinuity interfaces of a discontinuous function. This approach leverages Graph-Informed Neural Networks (GINNs) and sparse grids to address discontinuity detection also in domains of dimension larger than 3. GINNs, trained to identify troubled points on sparse grids, exploit graph structures built on the grids to achieve efficient and accurate discontinuity detection performances. We also introduce a recursive algorithm for general sparse grid-based detectors, characterized by convergence properties and easy applicability. Numerical experiments on functions with dimensions n = 2 and n = 4 demonstrate the efficiency and robust generalization of GINNs in detecting discontinuity interfaces. Notably, the trained GINNs offer portability and versatility, allowing integration into various algorithms and sharing among users.
    

