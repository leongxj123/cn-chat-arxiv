# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CEDAS: A Compressed Decentralized Stochastic Gradient Method with Improved Convergence](https://arxiv.org/abs/2301.05872) | CEDAS提出了一种压缩分布式随机梯度方法，在无偏压缩运算符下具有与集中式随机梯度下降相当的收敛速度，实现了最短的瞬态时间，对光滑强凸和非凸目标函数都适用。 |
| [^2] | [On the Effects of Data Heterogeneity on the Convergence Rates of Distributed Linear System Solvers.](http://arxiv.org/abs/2304.10640) | 本文比较了投影方法和优化方法求解分布式线性系统的收敛速度，提出了角异构性的几何概念，并对最有效的算法(APC和D-HBM)的收敛速度进行了约束和比较。 |

# 详细

[^1]: CEDAS：一种具有改进收敛性的压缩分布式随机梯度法

    CEDAS: A Compressed Decentralized Stochastic Gradient Method with Improved Convergence

    [https://arxiv.org/abs/2301.05872](https://arxiv.org/abs/2301.05872)

    CEDAS提出了一种压缩分布式随机梯度方法，在无偏压缩运算符下具有与集中式随机梯度下降相当的收敛速度，实现了最短的瞬态时间，对光滑强凸和非凸目标函数都适用。

    

    在本文中，我们考虑在通信受限环境下解决多代理网络上的分布式优化问题。我们研究了一种称为“具有自适应步长的压缩精确扩散（CEDAS）”的压缩分布式随机梯度方法，并证明该方法在无偏压缩运算符下渐近地实现了与集中式随机梯度下降（SGD）相当的收敛速度，适用于光滑强凸目标函数和光滑非凸目标函数。特别地，据我们所知，CEDAS迄今为止以其最短的瞬态时间（关于图的特性）实现了与集中式SGD相同的收敛速度，其在光滑强凸目标函数下表现为$\mathcal{O}(n{C^3}/(1-\lambda_2)^{2})$，在光滑非凸目标函数下表现为$\mathcal{O}(n^3{C^6}/(1-\lambda_2)^4)$，其中$(1-\lambda_2)$表示谱...

    arXiv:2301.05872v2 Announce Type: replace-cross  Abstract: In this paper, we consider solving the distributed optimization problem over a multi-agent network under the communication restricted setting. We study a compressed decentralized stochastic gradient method, termed ``compressed exact diffusion with adaptive stepsizes (CEDAS)", and show the method asymptotically achieves comparable convergence rate as centralized { stochastic gradient descent (SGD)} for both smooth strongly convex objective functions and smooth nonconvex objective functions under unbiased compression operators. In particular, to our knowledge, CEDAS enjoys so far the shortest transient time (with respect to the graph specifics) for achieving the convergence rate of centralized SGD, which behaves as $\mathcal{O}(n{C^3}/(1-\lambda_2)^{2})$ under smooth strongly convex objective functions, and $\mathcal{O}(n^3{C^6}/(1-\lambda_2)^4)$ under smooth nonconvex objective functions, where $(1-\lambda_2)$ denotes the spectr
    
[^2]: 论数据异构性对分布式线性系统求解器收敛速度的影响

    On the Effects of Data Heterogeneity on the Convergence Rates of Distributed Linear System Solvers. (arXiv:2304.10640v1 [cs.DC])

    [http://arxiv.org/abs/2304.10640](http://arxiv.org/abs/2304.10640)

    本文比较了投影方法和优化方法求解分布式线性系统的收敛速度，提出了角异构性的几何概念，并对最有效的算法(APC和D-HBM)的收敛速度进行了约束和比较。

    

    本文考虑了解决大规模线性方程组的基本问题。特别地，我们考虑任务负责人打算在一组具有一些方程组子集的机器的分布式/联合帮助下解决该系统的设置。虽然有几种方法用于解决这个问题，但缺少对投影方法和优化方法收敛速度的严格比较。在本文中，我们分析并比较这两类算法，特别关注每个类别中最有效的方法，即最近提出的加速投影一致性(APC)和分布式重球方法(D-HBM)。为此，我们首先提出了称为角异构性的几何概念，并讨论其普遍性。使用该概念，我们约束并比较所研究算法的收敛速度，并捕捉两种方法的异构数据的效应。

    We consider the fundamental problem of solving a large-scale system of linear equations. In particular, we consider the setting where a taskmaster intends to solve the system in a distributed/federated fashion with the help of a set of machines, who each have a subset of the equations. Although there exist several approaches for solving this problem, missing is a rigorous comparison between the convergence rates of the projection-based methods and those of the optimization-based ones. In this paper, we analyze and compare these two classes of algorithms with a particular focus on the most efficient method from each class, namely, the recently proposed Accelerated Projection-Based Consensus (APC) and the Distributed Heavy-Ball Method (D-HBM). To this end, we first propose a geometric notion of data heterogeneity called angular heterogeneity and discuss its generality. Using this notion, we bound and compare the convergence rates of the studied algorithms and capture the effects of both 
    

