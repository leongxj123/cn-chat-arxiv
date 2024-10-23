# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Driven Fixed-Point Tuning for Truncated Realized Variations.](http://arxiv.org/abs/2311.00905) | 本文提出了一种基于数据驱动的截断实现变异的固定点调整方法，有效估计积分波动性。 |
| [^2] | [Targeted Separation and Convergence with Kernel Discrepancies.](http://arxiv.org/abs/2209.12835) | 通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。 |

# 详细

[^1]: 数据驱动的截断实现变异的固定点调整方法

    Data-Driven Fixed-Point Tuning for Truncated Realized Variations. (arXiv:2311.00905v1 [math.ST])

    [http://arxiv.org/abs/2311.00905](http://arxiv.org/abs/2311.00905)

    本文提出了一种基于数据驱动的截断实现变异的固定点调整方法，有效估计积分波动性。

    

    在估计存在跳跃的半鞅的积分波动性和相关泛函时，许多方法需要指定调整参数的使用。在现有的理论中，调整参数被假设为确定性的，并且其值仅在渐近约束条件下指定。然而，在实证研究和模拟研究中，它们通常被选择为随机和数据相关的，实际上仅依赖于启发式方法。在本文中，我们考虑了一种基于一种随机固定点迭代的半鞅带跳跃的截断实现变异的新颖数据驱动调整程序。我们的方法是高度自动化的，可以减轻关于调整参数的微妙决策的需求，并且可以仅使用关于采样频率的信息进行实施。我们展示了我们的方法可以导致渐进有效的积分波动性估计，并展示了其在

    Many methods for estimating integrated volatility and related functionals of semimartingales in the presence of jumps require specification of tuning parameters for their use. In much of the available theory, tuning parameters are assumed to be deterministic, and their values are specified only up to asymptotic constraints. However, in empirical work and in simulation studies, they are typically chosen to be random and data-dependent, with explicit choices in practice relying on heuristics alone. In this paper, we consider novel data-driven tuning procedures for the truncated realized variations of a semimartingale with jumps, which are based on a type of stochastic fixed-point iteration. Being effectively automated, our approach alleviates the need for delicate decision-making regarding tuning parameters, and can be implemented using information regarding sampling frequency alone. We show our methods can lead to asymptotically efficient estimation of integrated volatility and exhibit 
    
[^2]: 通过核差异实现有针对性的分离与收敛

    Targeted Separation and Convergence with Kernel Discrepancies. (arXiv:2209.12835v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.12835](http://arxiv.org/abs/2209.12835)

    通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。

    

    最大均值差异（MMDs）如核Stein差异（KSD）已经成为广泛应用的中心，包括假设检验、采样器选择、分布近似和变分推断。在每个设置中，这些基于核的差异度量需要实现（i）将目标P与其他概率测度分离，甚至（ii）控制对P的弱收敛。在本文中，我们推导了确保（i）和（ii）的新的充分必要条件。对于可分的度量空间上的MMDs，我们描述了分离Bochner可嵌入测度的核，并引入简单的条件来分离所有具有无界核的测度和用有界核来控制收敛。我们利用这些结果在$\mathbb{R}^d$上大大扩展了KSD分离和收敛控制的已知条件，并开发了首个能够精确度量对P的弱收敛的KSDs。在这个过程中，我们强调了我们的结果的影响。

    Maximum mean discrepancies (MMDs) like the kernel Stein discrepancy (KSD) have grown central to a wide range of applications, including hypothesis testing, sampler selection, distribution approximation, and variational inference. In each setting, these kernel-based discrepancy measures are required to (i) separate a target P from other probability measures or even (ii) control weak convergence to P. In this article we derive new sufficient and necessary conditions to ensure (i) and (ii). For MMDs on separable metric spaces, we characterize those kernels that separate Bochner embeddable measures and introduce simple conditions for separating all measures with unbounded kernels and for controlling convergence with bounded kernels. We use these results on $\mathbb{R}^d$ to substantially broaden the known conditions for KSD separation and convergence control and to develop the first KSDs known to exactly metrize weak convergence to P. Along the way, we highlight the implications of our res
    

