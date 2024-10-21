# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the convergence of dynamic implementations of Hamiltonian Monte Carlo and No U-Turn Samplers.](http://arxiv.org/abs/2307.03460) | 本文研究了动态实现的Hamiltonian Monte Carlo (HMC)算法和No U-Turn Sampler (NUTS) 的收敛性，证明了NUTS作为动态HMC的特例，并且在一定条件下具有遍历性和几何遍历性。同时改进了HMC的收敛性结果，证明了在目标分布为高斯分布的微扰情况下，无需任何有界条件，HMC也是遍历的。 |
| [^2] | [Lower Complexity Adaptation for Empirical Entropic Optimal Transport.](http://arxiv.org/abs/2306.13580) | 本文研究了经验熵正则化最优输运的统计表现，并证明了它遵循低复杂度适应原则，推导出了其统计界限及参数化速率。 |

# 详细

[^1]: 动态实现的Hamiltonian Monte Carlo和No U-Turn Samplers的收敛性

    On the convergence of dynamic implementations of Hamiltonian Monte Carlo and No U-Turn Samplers. (arXiv:2307.03460v1 [stat.CO])

    [http://arxiv.org/abs/2307.03460](http://arxiv.org/abs/2307.03460)

    本文研究了动态实现的Hamiltonian Monte Carlo (HMC)算法和No U-Turn Sampler (NUTS) 的收敛性，证明了NUTS作为动态HMC的特例，并且在一定条件下具有遍历性和几何遍历性。同时改进了HMC的收敛性结果，证明了在目标分布为高斯分布的微扰情况下，无需任何有界条件，HMC也是遍历的。

    

    针对动态实现的Hamiltonian Monte Carlo (HMC)算法，例如No U-Turn Sampler (NUTS)，在许多具有挑战性的推理问题中具有成功的经验证据，但关于它们行为的理论结果还不足。本文旨在填补这一空白。具体而言，我们考虑了一个称为动态HMC的通用MCMC算法类。我们证明了这个通用框架涵盖了NUTS作为一个特例，并且作为一个附带结果，证明了目标分布的不变性。其次，我们建立了使NUTS不可约和非周期的条件，并作为推论而证明了遍历性。在类似于HMC的条件下，我们还证明了NUTS具有几何遍历性。最后，我们改进了现有的HMC收敛性结果，证明了这个方法在目标分布是高斯分布的微扰的情况下，无需对步长和leapfrog步数进行任何有界条件，也是遍历的。

    There is substantial empirical evidence about the success of dynamic implementations of Hamiltonian Monte Carlo (HMC), such as the No U-Turn Sampler (NUTS), in many challenging inference problems but theoretical results about their behavior are scarce. The aim of this paper is to fill this gap. More precisely, we consider a general class of MCMC algorithms we call dynamic HMC. We show that this general framework encompasses NUTS as a particular case, implying the invariance of the target distribution as a by-product. Second, we establish conditions under which NUTS is irreducible and aperiodic and as a corrolary ergodic. Under conditions similar to the ones existing for HMC, we also show that NUTS is geometrically ergodic. Finally, we improve existing convergence results for HMC showing that this method is ergodic without any boundedness condition on the stepsize and the number of leapfrog steps, in the case where the target is a perturbation of a Gaussian distribution.
    
[^2]: 经验熵正则化最优输运的低复杂度适应性

    Lower Complexity Adaptation for Empirical Entropic Optimal Transport. (arXiv:2306.13580v1 [math.ST])

    [http://arxiv.org/abs/2306.13580](http://arxiv.org/abs/2306.13580)

    本文研究了经验熵正则化最优输运的统计表现，并证明了它遵循低复杂度适应原则，推导出了其统计界限及参数化速率。

    

    经验熵正则化最优输运 (EOT) 是优化输运 (OT) 的一种有效且计算可行的替代方案，对大规模数据分析有着广泛的应用。本文推导出了 EOT 成本的新的统计界限，并显示它们在熵正则化参数 $\epsilon$ 和样本大小 $n$ 的统计性能仅取决于两个概率测度之中较简单的那个。例如，在充分平滑的成本下，这会产生具有$\epsilon^{-d/2}$因子的参数化速率$n^{-1/2}$，其中$d$是两个总体测度的最小维度。这确认了经验EOT也遵循了最近才为未规则化OT确认的低复杂度适应原则的标志性特征。根据我们的理论，我们展示了欧几里得空间上的测度的经验熵Gromov-Wasserstein距离及其未规则化版本也遵循此原则。

    Entropic optimal transport (EOT) presents an effective and computationally viable alternative to unregularized optimal transport (OT), offering diverse applications for large-scale data analysis. In this work, we derive novel statistical bounds for empirical plug-in estimators of the EOT cost and show that their statistical performance in the entropy regularization parameter $\epsilon$ and the sample size $n$ only depends on the simpler of the two probability measures. For instance, under sufficiently smooth costs this yields the parametric rate $n^{-1/2}$ with factor $\epsilon^{-d/2}$, where $d$ is the minimum dimension of the two population measures. This confirms that empirical EOT also adheres to the lower complexity adaptation principle, a hallmark feature only recently identified for unregularized OT. As a consequence of our theory, we show that the empirical entropic Gromov-Wasserstein distance and its unregularized version for measures on Euclidean spaces also obey this princip
    

