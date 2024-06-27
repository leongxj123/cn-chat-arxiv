# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MgNO: Efficient Parameterization of Linear Operators via Multigrid.](http://arxiv.org/abs/2310.19809) | 本文提出了一个简洁的神经算子架构，通过多重网格结构有效参数化线性算子，实现了算子学习的数学严密性和实用性。 |

# 详细

[^1]: MgNO:通过多重网格有效参数化线性算子

    MgNO: Efficient Parameterization of Linear Operators via Multigrid. (arXiv:2310.19809v1 [cs.LG])

    [http://arxiv.org/abs/2310.19809](http://arxiv.org/abs/2310.19809)

    本文提出了一个简洁的神经算子架构，通过多重网格结构有效参数化线性算子，实现了算子学习的数学严密性和实用性。

    

    本文提出了一个简洁的神经算子架构来进行算子学习。将其与传统的全连接神经网络进行类比，将神经算子定义为非线性算子层中第$i$个神经元的输出，记作$\mathcal O_i(u) = \sigma\left( \sum_j \mathcal W_{ij} u + \mathcal B_{ij}\right)$。其中，$\mathcal W_{ij}$表示连接第$j$个输入神经元和第$i$个输出神经元的有界线性算子，而偏差$\mathcal B_{ij}$采用函数形式而非标量形式。通过在两个神经元（Banach空间）之间有效参数化有界线性算子，MgNO引入了多重网格结构。这种方法既具备了数学严密性，又具备了实用性。此外，MgNO消除了对传统的lifting和projecting操作的需求。

    In this work, we propose a concise neural operator architecture for operator learning. Drawing an analogy with a conventional fully connected neural network, we define the neural operator as follows: the output of the $i$-th neuron in a nonlinear operator layer is defined by $\mathcal O_i(u) = \sigma\left( \sum_j \mathcal W_{ij} u + \mathcal B_{ij}\right)$. Here, $\mathcal W_{ij}$ denotes the bounded linear operator connecting $j$-th input neuron to $i$-th output neuron, and the bias $\mathcal B_{ij}$ takes the form of a function rather than a scalar. Given its new universal approximation property, the efficient parameterization of the bounded linear operators between two neurons (Banach spaces) plays a critical role. As a result, we introduce MgNO, utilizing multigrid structures to parameterize these linear operators between neurons. This approach offers both mathematical rigor and practical expressivity. Additionally, MgNO obviates the need for conventional lifting and projecting ope
    

