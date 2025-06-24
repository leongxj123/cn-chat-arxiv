# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LieDetect: Detection of representation orbits of compact Lie groups from point clouds.](http://arxiv.org/abs/2309.03086) | LieDetect是一种从紧致Lie群的有限样本轨道中估计表示的新算法。与其他技术不同，该算法可以检索精确的表示类型，并重建其轨道，有助于识别生成该作用的Lie群。该算法适用于任何紧致Lie群，并在多个领域的应用中取得了非常准确的结果。 |
| [^2] | [On the Effects of Data Heterogeneity on the Convergence Rates of Distributed Linear System Solvers.](http://arxiv.org/abs/2304.10640) | 本文比较了投影方法和优化方法求解分布式线性系统的收敛速度，提出了角异构性的几何概念，并对最有效的算法(APC和D-HBM)的收敛速度进行了约束和比较。 |
| [^3] | [On the fast convergence of minibatch heavy ball momentum.](http://arxiv.org/abs/2206.07553) | 本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。 |

# 详细

[^1]: LieDetect: 从点云中检测紧致Lie群的表示轨道

    LieDetect: Detection of representation orbits of compact Lie groups from point clouds. (arXiv:2309.03086v1 [math.OC])

    [http://arxiv.org/abs/2309.03086](http://arxiv.org/abs/2309.03086)

    LieDetect是一种从紧致Lie群的有限样本轨道中估计表示的新算法。与其他技术不同，该算法可以检索精确的表示类型，并重建其轨道，有助于识别生成该作用的Lie群。该算法适用于任何紧致Lie群，并在多个领域的应用中取得了非常准确的结果。

    

    我们提出了一种新的算法，用于从紧致Lie群的有限样本轨道中估计表示。与其他报道的技术不同，我们的方法允许检索精确的表示类型，作为不可约表示的直和。而且，对表示类型的了解可以重建其轨道，有助于识别生成该作用的Lie群。我们的算法适用于任何紧致Lie群，但只考虑了SO(2), T^d, SU(2)和SO(3)的实例化。我们推导了在Hausdorff和Wasserstein距离方面的鲁棒性的理论保证。我们的工具来自于几何测度理论，计算几何和矩阵流形上的优化。算法在高达16维的合成数据以及图像分析，谐波分析和经典力学系统的实际应用中进行了测试，取得了非常准确的结果。

    We suggest a new algorithm to estimate representations of compact Lie groups from finite samples of their orbits. Different from other reported techniques, our method allows the retrieval of the precise representation type as a direct sum of irreducible representations. Moreover, the knowledge of the representation type permits the reconstruction of its orbit, which is useful to identify the Lie group that generates the action. Our algorithm is general for any compact Lie group, but only instantiations for SO(2), T^d, SU(2) and SO(3) are considered. Theoretical guarantees of robustness in terms of Hausdorff and Wasserstein distances are derived. Our tools are drawn from geometric measure theory, computational geometry, and optimization on matrix manifolds. The algorithm is tested for synthetic data up to dimension 16, as well as real-life applications in image analysis, harmonic analysis, and classical mechanics systems, achieving very accurate results.
    
[^2]: 论数据异构性对分布式线性系统求解器收敛速度的影响

    On the Effects of Data Heterogeneity on the Convergence Rates of Distributed Linear System Solvers. (arXiv:2304.10640v1 [cs.DC])

    [http://arxiv.org/abs/2304.10640](http://arxiv.org/abs/2304.10640)

    本文比较了投影方法和优化方法求解分布式线性系统的收敛速度，提出了角异构性的几何概念，并对最有效的算法(APC和D-HBM)的收敛速度进行了约束和比较。

    

    本文考虑了解决大规模线性方程组的基本问题。特别地，我们考虑任务负责人打算在一组具有一些方程组子集的机器的分布式/联合帮助下解决该系统的设置。虽然有几种方法用于解决这个问题，但缺少对投影方法和优化方法收敛速度的严格比较。在本文中，我们分析并比较这两类算法，特别关注每个类别中最有效的方法，即最近提出的加速投影一致性(APC)和分布式重球方法(D-HBM)。为此，我们首先提出了称为角异构性的几何概念，并讨论其普遍性。使用该概念，我们约束并比较所研究算法的收敛速度，并捕捉两种方法的异构数据的效应。

    We consider the fundamental problem of solving a large-scale system of linear equations. In particular, we consider the setting where a taskmaster intends to solve the system in a distributed/federated fashion with the help of a set of machines, who each have a subset of the equations. Although there exist several approaches for solving this problem, missing is a rigorous comparison between the convergence rates of the projection-based methods and those of the optimization-based ones. In this paper, we analyze and compare these two classes of algorithms with a particular focus on the most efficient method from each class, namely, the recently proposed Accelerated Projection-Based Consensus (APC) and the Distributed Heavy-Ball Method (D-HBM). To this end, we first propose a geometric notion of data heterogeneity called angular heterogeneity and discuss its generality. Using this notion, we bound and compare the convergence rates of the studied algorithms and capture the effects of both 
    
[^3]: 论小批量重球动量法的快速收敛性

    On the fast convergence of minibatch heavy ball momentum. (arXiv:2206.07553v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.07553](http://arxiv.org/abs/2206.07553)

    本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。

    

    简单的随机动量方法被广泛用于机器学习优化中，但由于还没有加速的理论保证，这与它们在实践中的良好性能并不相符。本文旨在通过展示，随机重球动量在二次最优化问题中保持（确定性）重球动量的快速线性率，至少在使用足够大的批量大小进行小批量处理时。我们所研究的算法可以被解释为带小批量处理和重球动量的加速随机Kaczmarz算法。该分析依赖于仔细分解动量转移矩阵，并使用新的独立随机矩阵乘积的谱范围集中界限。我们提供了数值演示，证明了我们的界限相当尖锐。

    Simple stochastic momentum methods are widely used in machine learning optimization, but their good practical performance is at odds with an absence of theoretical guarantees of acceleration in the literature. In this work, we aim to close the gap between theory and practice by showing that stochastic heavy ball momentum retains the fast linear rate of (deterministic) heavy ball momentum on quadratic optimization problems, at least when minibatching with a sufficiently large batch size. The algorithm we study can be interpreted as an accelerated randomized Kaczmarz algorithm with minibatching and heavy ball momentum. The analysis relies on carefully decomposing the momentum transition matrix, and using new spectral norm concentration bounds for products of independent random matrices. We provide numerical illustrations demonstrating that our bounds are reasonably sharp.
    

