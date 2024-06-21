# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning-based Multi-continuum Model for Multiscale Flow Problems](https://arxiv.org/abs/2403.14084) | 提出了一种基于学习的多连续体模型，用于改进多尺度问题中单一连续体模型的准确性 |
| [^2] | [DynGMA: a robust approach for learning stochastic differential equations from data](https://arxiv.org/abs/2402.14475) | DynGMA方法通过引入新的密度近似，优于基准方法在学习完全未知的漂移和扩散函数以及计算不变性方面的准确性。 |
| [^3] | [Diffeomorphism Neural Operator for various domains and parameters of partial differential equations](https://arxiv.org/abs/2402.12475) | 通过微分同胚神经算子学习框架，提出了一种适用于各种和复杂领域的物理系统的领域灵活模型，从而将学习函数映射在不同领域的问题转化为在共享的微分同胚上学习算子的问题。 |

# 详细

[^1]: 基于学习的多孔介质模型用于多尺度流动问题

    Learning-based Multi-continuum Model for Multiscale Flow Problems

    [https://arxiv.org/abs/2403.14084](https://arxiv.org/abs/2403.14084)

    提出了一种基于学习的多连续体模型，用于改进多尺度问题中单一连续体模型的准确性

    

    多尺度问题通常可以通过数值均质化来近似，通过具有某些有效参数的方程来捕获原始系统在粗网格上的宏观行为，以加快模拟速度。然而，这种方法通常假设尺度分离，并且解的异质性可以通过每个粗块中的解的平均值来近似。对于复杂的多尺度问题，计算的单一有效性特性/连续体可能不足够。在本文中，我们提出了一种新颖的基于学习的多连续体模型，用于丰富均质化方程并提高多尺度问题单一连续体模型的准确性，给定一些数据。不失一般性，我们考虑了一个双连续体的情况。第一个流动方程保留了原始均质化方程的信息，具有额外的交互项。第二个连续体是新引入的。

    arXiv:2403.14084v1 Announce Type: cross  Abstract: Multiscale problems can usually be approximated through numerical homogenization by an equation with some effective parameters that can capture the macroscopic behavior of the original system on the coarse grid to speed up the simulation. However, this approach usually assumes scale separation and that the heterogeneity of the solution can be approximated by the solution average in each coarse block. For complex multiscale problems, the computed single effective properties/continuum might be inadequate. In this paper, we propose a novel learning-based multi-continuum model to enrich the homogenized equation and improve the accuracy of the single continuum model for multiscale problems with some given data. Without loss of generalization, we consider a two-continuum case. The first flow equation keeps the information of the original homogenized equation with an additional interaction term. The second continuum is newly introduced, and t
    
[^2]: DynGMA：一种从数据学习随机微分方程的稳健方法

    DynGMA: a robust approach for learning stochastic differential equations from data

    [https://arxiv.org/abs/2402.14475](https://arxiv.org/abs/2402.14475)

    DynGMA方法通过引入新的密度近似，优于基准方法在学习完全未知的漂移和扩散函数以及计算不变性方面的准确性。

    

    从观测数据中学习未知的随机微分方程（SDEs）是一项重要且具有挑战性的任务，应用于各个领域。本文引入了新的近似参数化SDE转移密度的方法：受动力系统随机摄动理论启发的高斯密度近似，以及它的扩展，动力高斯混合近似（DynGMA）。受益于稳健的密度近似，我们的方法在学习完全未知的漂移和扩散函数以及计算矩不变性方面表现出优越的准确性。

    arXiv:2402.14475v1 Announce Type: new  Abstract: Learning unknown stochastic differential equations (SDEs) from observed data is a significant and challenging task with applications in various fields. Current approaches often use neural networks to represent drift and diffusion functions, and construct likelihood-based loss by approximating the transition density to train these networks. However, these methods often rely on one-step stochastic numerical schemes, necessitating data with sufficiently high time resolution. In this paper, we introduce novel approximations to the transition density of the parameterized SDE: a Gaussian density approximation inspired by the random perturbation theory of dynamical systems, and its extension, the dynamical Gaussian mixture approximation (DynGMA). Benefiting from the robust density approximation, our method exhibits superior accuracy compared to baseline methods in learning the fully unknown drift and diffusion functions and computing the invari
    
[^3]: 不同领域和参数的微分同胚神经算子

    Diffeomorphism Neural Operator for various domains and parameters of partial differential equations

    [https://arxiv.org/abs/2402.12475](https://arxiv.org/abs/2402.12475)

    通过微分同胚神经算子学习框架，提出了一种适用于各种和复杂领域的物理系统的领域灵活模型，从而将学习函数映射在不同领域的问题转化为在共享的微分同胚上学习算子的问题。

    

    许多科学和工程应用需要对传统上使用资源密集型数值求解器计算的偏微分方程（PDE）进行评估。神经算子模型通过直接从数据中学习控制物理定律，提供了一种有效的替代方案，适用于具有不同参数的PDE类别，但在固定边界（领域）内受限。许多应用，例如设计和制造，在大规模研究时将受益于具有灵活领域的神经算子。在这里，我们提出了一种微分同胚神经算子学习框架，旨在为具有各种和复杂领域的物理系统开发领域灵活模型。具体来说，提出了一个在由微分同胚从各领域映射而来的共享领域中训练的神经算子，该方法将在不同领域（空间）学习函数映射的问题转化为在共享的微分同胚上学习算子的问题。

    arXiv:2402.12475v1 Announce Type: cross  Abstract: Many science and engineering applications demand partial differential equations (PDE) evaluations that are traditionally computed with resource-intensive numerical solvers. Neural operator models provide an efficient alternative by learning the governing physical laws directly from data in a class of PDEs with different parameters, but constrained in a fixed boundary (domain). Many applications, such as design and manufacturing, would benefit from neural operators with flexible domains when studied at scale. Here we present a diffeomorphism neural operator learning framework towards developing domain-flexible models for physical systems with various and complex domains. Specifically, a neural operator trained in a shared domain mapped from various domains of fields by diffeomorphism is proposed, which transformed the problem of learning function mappings in varying domains (spaces) into the problem of learning operators on a shared dif
    

