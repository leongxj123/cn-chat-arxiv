# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-dependent PDE solver based on signature kernels](https://arxiv.org/abs/2403.11738) | 该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。 |
| [^2] | [A Note on Randomized Kaczmarz Algorithm for Solving Doubly-Noisy Linear Systems.](http://arxiv.org/abs/2308.16904) | 本文分析了当系数矩阵和向量都存在加性和乘性噪声时，随机Kaczmarz算法在解决噪声线性系统中的收敛性。分析表明，RK的收敛性受到𝜏的大小影响，其中𝜏表示带有噪声的系数矩阵A的乘子范数的平方与Frobenius范数的平方的乘积。 |

# 详细

[^1]: 基于特征核的路径依赖PDE求解器

    A path-dependent PDE solver based on signature kernels

    [https://arxiv.org/abs/2403.11738](https://arxiv.org/abs/2403.11738)

    该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。

    

    我们开发了一种基于特征核的路径依赖PDE（PPDE）的收敛证明求解器。我们的数值方案利用了特征核，这是最近在路径空间上引入的一类核。具体来说，我们通过在符号再生核希尔伯特空间（RKHS）中近似PPDE的解来解决一个最优恢复问题，该空间受到在有限集合的远程路径上满足PPDE约束的元素的约束。在线性情况下，我们证明了优化具有唯一的闭式解，其以远程路径处的特征核评估的形式表示。我们证明了所提出方案的一致性，保证在远程点数增加时收敛到PPDE解。最后，我们提供了几个数值例子，尤其是在粗糙波动率下的期权定价背景下。我们的数值方案构成了一种替代性的蒙特卡洛方法的有效替代方案。

    arXiv:2403.11738v1 Announce Type: cross  Abstract: We develop a provably convergent kernel-based solver for path-dependent PDEs (PPDEs). Our numerical scheme leverages signature kernels, a recently introduced class of kernels on path-space. Specifically, we solve an optimal recovery problem by approximating the solution of a PPDE with an element of minimal norm in the signature reproducing kernel Hilbert space (RKHS) constrained to satisfy the PPDE at a finite collection of collocation paths. In the linear case, we show that the optimisation has a unique closed-form solution expressed in terms of signature kernel evaluations at the collocation paths. We prove consistency of the proposed scheme, guaranteeing convergence to the PPDE solution as the number of collocation points increases. Finally, several numerical examples are presented, in particular in the context of option pricing under rough volatility. Our numerical scheme constitutes a valid alternative to the ubiquitous Monte Carl
    
[^2]: 关于解决双向噪声线性系统的随机Kaczmarz算法的注释

    A Note on Randomized Kaczmarz Algorithm for Solving Doubly-Noisy Linear Systems. (arXiv:2308.16904v1 [math.NA])

    [http://arxiv.org/abs/2308.16904](http://arxiv.org/abs/2308.16904)

    本文分析了当系数矩阵和向量都存在加性和乘性噪声时，随机Kaczmarz算法在解决噪声线性系统中的收敛性。分析表明，RK的收敛性受到𝜏的大小影响，其中𝜏表示带有噪声的系数矩阵A的乘子范数的平方与Frobenius范数的平方的乘积。

    

    大规模线性系统Ax=b在实践中经常出现，需要有效的迭代求解器。通常，由于操作误差或错误的数据收集过程，这些系统会出现噪声。在过去的十年中，随机Kaczmarz（RK）算法已被广泛研究作为这些系统的高效迭代求解器。然而，现有对RK在噪声情况下的收敛性研究有限，只考虑右侧向量b中的测量噪声。不幸的是，在实践中，并不总是这样；系数矩阵A也可能是有噪声的。在本文中，我们分析了当系数矩阵A以及向量b都受有加性和乘性噪声影响时，RK的收敛性。在我们的分析中，变量 𝜏=∥ 𝜏 𝐴 ∗ ∥2^2 ∥ 𝐜𝐷𝐻∗𝑐 − 𝐛 ∥_𝐹^2🈶 𝑜 𝑅 的大小会影响RK的收敛性，其中 𝜏𝐴 表示A的带有噪声的版本。我们声称我们的分析是健壮且逼近实际的.

    Large-scale linear systems, $Ax=b$, frequently arise in practice and demand effective iterative solvers. Often, these systems are noisy due to operational errors or faulty data-collection processes. In the past decade, the randomized Kaczmarz (RK) algorithm has been studied extensively as an efficient iterative solver for such systems. However, the convergence study of RK in the noisy regime is limited and considers measurement noise in the right-hand side vector, $b$. Unfortunately, in practice, that is not always the case; the coefficient matrix $A$ can also be noisy. In this paper, we analyze the convergence of RK for noisy linear systems when the coefficient matrix, $A$, is corrupted with both additive and multiplicative noise, along with the noisy vector, $b$. In our analyses, the quantity $\tilde R=\| \tilde A^{\dagger} \|_2^2 \|\tilde A \|_F^2$ influences the convergence of RK, where $\tilde A$ represents a noisy version of $A$. We claim that our analysis is robust and realistic
    

