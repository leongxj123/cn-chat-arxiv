# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Parallel-in-Time Probabilistic Numerical ODE Solvers.](http://arxiv.org/abs/2310.01145) | 本文提出了一种并行时间概率数值ODE求解器，通过将数值模拟问题视为贝叶斯状态估计问题，并利用贝叶斯滤波和平滑的框架，实现了在并行处理所有时间步骤的同时将时间开销降低到对数级别。 |
| [^2] | [Neural Integral Equations.](http://arxiv.org/abs/2209.15190) | 本文介绍了神经积分方程（NIE）和自注意神经积分方程（ANIE）的方法，它们可以在无监督情况下通过学习数据中的积分算子进行模型建立，并且在合成和真实世界数据的ODE、PDE和IE系统中的基准任务上表现出较高的速度和准确性。 |

# 详细

[^1]: 并行时间概率数值ODE求解器

    Parallel-in-Time Probabilistic Numerical ODE Solvers. (arXiv:2310.01145v1 [math.NA])

    [http://arxiv.org/abs/2310.01145](http://arxiv.org/abs/2310.01145)

    本文提出了一种并行时间概率数值ODE求解器，通过将数值模拟问题视为贝叶斯状态估计问题，并利用贝叶斯滤波和平滑的框架，实现了在并行处理所有时间步骤的同时将时间开销降低到对数级别。

    

    针对常微分方程(ODE)的概率数值求解器将动力系统的数值仿真问题视为贝叶斯状态估计问题。除了生成ODE解的后验分布并因此量化方法本身的数值逼近误差之外，这种形式化方法的一个不常被注意到的优势是通过在贝叶斯滤波和平滑的框架中进行数值模拟而获得的算法灵活性。在本文中，我们利用这种灵活性，基于时间并行迭代扩展卡尔曼平滑器的公式化，提出了一种并行时间概率数值ODE求解器。与当前的概率求解器依次按时间顺序模拟动力系统不同，所提出的方法以并行方式处理所有时间步骤，从而将时间开销从线性降低到对数级别的时间步骤数。我们通过在多种问题上展示了我们方法的有效性。

    Probabilistic numerical solvers for ordinary differential equations (ODEs) treat the numerical simulation of dynamical systems as problems of Bayesian state estimation. Aside from producing posterior distributions over ODE solutions and thereby quantifying the numerical approximation error of the method itself, one less-often noted advantage of this formalism is the algorithmic flexibility gained by formulating numerical simulation in the framework of Bayesian filtering and smoothing. In this paper, we leverage this flexibility and build on the time-parallel formulation of iterated extended Kalman smoothers to formulate a parallel-in-time probabilistic numerical ODE solver. Instead of simulating the dynamical system sequentially in time, as done by current probabilistic solvers, the proposed method processes all time steps in parallel and thereby reduces the span cost from linear to logarithmic in the number of time steps. We demonstrate the effectiveness of our approach on a variety o
    
[^2]: 神经积分方程

    Neural Integral Equations. (arXiv:2209.15190v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15190](http://arxiv.org/abs/2209.15190)

    本文介绍了神经积分方程（NIE）和自注意神经积分方程（ANIE）的方法，它们可以在无监督情况下通过学习数据中的积分算子进行模型建立，并且在合成和真实世界数据的ODE、PDE和IE系统中的基准任务上表现出较高的速度和准确性。

    

    积分方程 (IEs) 是用于建模具有非局部相互作用的时空系统的方程。它们已经在理论和应用科学中找到了重要应用，包括物理学、化学、生物学和工程学。虽然存在有效的算法来解决给定的IEs，但不存在可以仅从数据中学习IE和其相关动态的方法。在本文中，我们介绍了神经积分方程 (NIE)，这种方法通过IE求解器从数据中学习未知的积分算子。我们还介绍了自注意神经积分方程 (ANIE)，其中积分被自注意力替换，这提高了可扩展性、容量，并产生了一个可解释的模型。我们证明(A)NIE在ODE、PDE和IE系统中的多个基准任务上的速度和准确性都优于其他方法，并且适用于合成和真实世界数据。

    Integral equations (IEs) are equations that model spatiotemporal systems with non-local interactions. They have found important applications throughout theoretical and applied sciences, including in physics, chemistry, biology, and engineering. While efficient algorithms exist for solving given IEs, no method exists that can learn an IE and its associated dynamics from data alone. In this paper, we introduce Neural Integral Equations (NIE), a method that learns an unknown integral operator from data through an IE solver. We also introduce Attentional Neural Integral Equations (ANIE), where the integral is replaced by self-attention, which improves scalability, capacity, and results in an interpretable model. We demonstrate that (A)NIE outperforms other methods in both speed and accuracy on several benchmark tasks in ODE, PDE, and IE systems of synthetic and real-world data.
    

