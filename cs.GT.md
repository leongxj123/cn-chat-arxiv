# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Exponentially Converging Particle Method for the Mixed Nash Equilibrium of Continuous Games.](http://arxiv.org/abs/2211.01280) | 本文提出并分析了一种基于粒子的方法，用于计算具有连续纯策略集和对收益函数的一阶访问的两人零和博弈的混合纳什均衡问题，并在满足假设的情况下从任何初始化指数收敛于准确的解。 |

# 详细

[^1]: 一种连续博弈混合纳什均衡的指数收敛粒子方法

    An Exponentially Converging Particle Method for the Mixed Nash Equilibrium of Continuous Games. (arXiv:2211.01280v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2211.01280](http://arxiv.org/abs/2211.01280)

    本文提出并分析了一种基于粒子的方法，用于计算具有连续纯策略集和对收益函数的一阶访问的两人零和博弈的混合纳什均衡问题，并在满足假设的情况下从任何初始化指数收敛于准确的解。

    

    本文考虑解决具有连续纯策略集和对收益函数的一阶访问的两人零和博弈的混合纳什均衡计算问题。该问题在以博弈理论为灵感的机器学习应用中出现，如分布式稳健学习。在这些应用中，策略集是高维的，因此基于离散化的方法不能返回高精度的解。本文引入并分析了一种基于粒子的方法，该方法针对此问题具有保证的局部收敛性。该方法将混合策略参数化为原子测度，并对原子的权重和位置应用近端点更新。它可以被解释为“相互作用”Wasserstein-Fisher-Rao梯度流的时间隐式离散化。我们证明，在非退化的假设下，该方法从任何初始化以指数速度收敛于准确的混合纳什均衡，并提供数值实验来说明该方法的实际性能。

    We consider the problem of computing mixed Nash equilibria of two-player zero-sum games with continuous sets of pure strategies and with first-order access to the payoff function. This problem arises for example in game-theory-inspired machine learning applications, such as distributionally-robust learning. In those applications, the strategy sets are high-dimensional and thus methods based on discretisation cannot tractably return high-accuracy solutions.  In this paper, we introduce and analyze a particle-based method that enjoys guaranteed local convergence for this problem. This method consists in parametrizing the mixed strategies as atomic measures and applying proximal point updates to both the atoms' weights and positions. It can be interpreted as a time-implicit discretization of the "interacting" Wasserstein-Fisher-Rao gradient flow.  We prove that, under non-degeneracy assumptions, this method converges at an exponential rate to the exact mixed Nash equilibrium from any init
    

