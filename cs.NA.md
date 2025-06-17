# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mitigating distribution shift in machine learning-augmented hybrid simulation.](http://arxiv.org/abs/2401.09259) | 本文研究了机器学习增强的混合模拟中的分布偏移问题，并提出了基于切线空间正则化估计器的方法来控制分布偏移，从而提高模拟结果的精确性。 |

# 详细

[^1]: 缓解机器学习增强的混合模拟中的分布偏移

    Mitigating distribution shift in machine learning-augmented hybrid simulation. (arXiv:2401.09259v1 [math.NA])

    [http://arxiv.org/abs/2401.09259](http://arxiv.org/abs/2401.09259)

    本文研究了机器学习增强的混合模拟中的分布偏移问题，并提出了基于切线空间正则化估计器的方法来控制分布偏移，从而提高模拟结果的精确性。

    

    本文研究了机器学习增强的混合模拟中普遍存在的分布偏移问题，其中模拟算法的部分被数据驱动的替代模型取代。我们首先建立了一个数学框架来理解机器学习增强的混合模拟问题的结构，以及相关的分布偏移的原因和影响。我们在数值和理论上展示了分布偏移与模拟误差的相关性。然后，我们提出了一种基于切线空间正则化估计器的简单方法来控制分布偏移，从而提高模拟结果的长期精确性。在线性动力学情况下，我们提供了一种详尽的理论分析来量化所提出方法的有效性。此外，我们进行了几个数值实验，包括模拟部分已知的反应扩散方程以及使用基于数据驱动的投影方法求解Navier-Stokes方程。

    We study the problem of distribution shift generally arising in machine-learning augmented hybrid simulation, where parts of simulation algorithms are replaced by data-driven surrogates. We first establish a mathematical framework to understand the structure of machine-learning augmented hybrid simulation problems, and the cause and effect of the associated distribution shift. We show correlations between distribution shift and simulation error both numerically and theoretically. Then, we propose a simple methodology based on tangent-space regularized estimator to control the distribution shift, thereby improving the long-term accuracy of the simulation results. In the linear dynamics case, we provide a thorough theoretical analysis to quantify the effectiveness of the proposed method. Moreover, we conduct several numerical experiments, including simulating a partially known reaction-diffusion equation and solving Navier-Stokes equations using the projection method with a data-driven p
    

