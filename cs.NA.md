# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep polytopic autoencoders for low-dimensional linear parameter-varying approximations and nonlinear feedback design](https://arxiv.org/abs/2403.18044) | 该研究开发了一种用于控制应用的深度多面体自编码器，在大规模系统的计算非线性控制器设计中展现出比标准线性方法更好的性能，其特定架构使得实现更高阶级数展开几乎没有额外计算负担。 |
| [^2] | [Models for information propagation on graphs.](http://arxiv.org/abs/2201.07577) | 本文提出了统一的图上信息传播模型，其中包括三种不同的类别，利用波、路径行程时间和eikonal方程来描述信息的传播，并给出了它们之间的等价性。此外，本文还提出了一种新的混合模型，用于描述波和eikonal模型的结合。作者在随机图形、小世界图和实际网络上进行了数值模拟。 |

# 详细

[^1]: 深度多面体自编码器用于低维线性参数变化逼近和非线性反馈设计

    Deep polytopic autoencoders for low-dimensional linear parameter-varying approximations and nonlinear feedback design

    [https://arxiv.org/abs/2403.18044](https://arxiv.org/abs/2403.18044)

    该研究开发了一种用于控制应用的深度多面体自编码器，在大规模系统的计算非线性控制器设计中展现出比标准线性方法更好的性能，其特定架构使得实现更高阶级数展开几乎没有额外计算负担。

    

    多面体自编码器提供了多面体中状态的低维参数化。对于非线性PDE，这很容易应用于低维线性参数变化(LPV)逼近，因为它们已被用于通过状态相关Riccati方程的级数展开实现有效的非线性控制器设计。在这项工作中，我们开发了一种用于控制应用的多面体自编码器，并展示了它如何在视图非线性系统的LPV逼近方面优于标准线性方法，以及特定架构如何在几乎没有额外计算的情况下实现更高阶级数展开。我们通过彻底的数值研究展示了该方法在大规模系统的计算非线性控制器设计中的性质和潜力。

    arXiv:2403.18044v1 Announce Type: cross  Abstract: Polytopic autoencoders provide low-dimensional parametrizations of states in a polytope. For nonlinear PDEs, this is readily applied to low-dimensional linear parameter-varying (LPV) approximations as they have been exploited for efficient nonlinear controller design via series expansions of the solution to the state-dependent Riccati equation. In this work, we develop a polytopic autoencoder for control applications and show how it outperforms standard linear approaches in view of LPV approximations of nonlinear systems and how the particular architecture enables higher order series expansions at little extra computational effort. We illustrate the properties and potentials of this approach to computational nonlinear controller design for large-scale systems with a thorough numerical study.
    
[^2]: 图上信息传播模型

    Models for information propagation on graphs. (arXiv:2201.07577v3 [math.NA] UPDATED)

    [http://arxiv.org/abs/2201.07577](http://arxiv.org/abs/2201.07577)

    本文提出了统一的图上信息传播模型，其中包括三种不同的类别，利用波、路径行程时间和eikonal方程来描述信息的传播，并给出了它们之间的等价性。此外，本文还提出了一种新的混合模型，用于描述波和eikonal模型的结合。作者在随机图形、小世界图和实际网络上进行了数值模拟。

    

    我们提出和统一了不同的图上信息传播模型。第一类模型将传播建模为一种波，它从一组已知节点在初始时间开始向所有其他未知节点传播，传播的顺序由信息波前的到达时间确定。第二类模型基于路径上的行程时间的概念。从一组初始已知节点到一个节点的信息传播时间被定义为所有可以到达该节点的路径的子集上的广义旅行时间的最小值。最后一个模型类是通过在每个未知节点上施加一个eikonal形式的局部方程，并在已知节点处施加边界条件来给出的。在一个节点的解的值与具有较低值的相邻节点的解的值耦合。我们提供了模型类的精确公式，并证明了它们之间的等价性。受到第一到达时间模型和eikonal方程之间的联系的启发，我们提出了一种新的混合形式，结合了波和eikonal模型。最后，我们在各种图形上展示了模型的数值模拟，包括随机图形、小世界图和实际网络。

    We propose and unify classes of different models for information propagation over graphs. In a first class, propagation is modelled as a wave which emanates from a set of known nodes at an initial time, to all other unknown nodes at later times with an ordering determined by the arrival time of the information wave front. A second class of models is based on the notion of a travel time along paths between nodes. The time of information propagation from an initial known set of nodes to a node is defined as the minimum of a generalised travel time over subsets of all admissible paths. A final class is given by imposing a local equation of an eikonal form at each unknown node, with boundary conditions at the known nodes. The solution value of the local equation at a node is coupled to those of neighbouring nodes with lower values. We provide precise formulations of the model classes and prove equivalences between them. Motivated by the connection between first arrival time model and the e
    

