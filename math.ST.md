# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Detection of Correlated Random Vectors.](http://arxiv.org/abs/2401.13429) | 本文研究了判断两个标准正态随机向量是否相关的问题，提出了一种新的方法来评估似然比的二阶矩，并发现了与整数分割函数之间的联系。 |
| [^2] | [Nonparametric extensions of randomized response for private confidence sets.](http://arxiv.org/abs/2202.08728) | 本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。 |

# 详细

[^1]: 相关随机向量的检测

    Detection of Correlated Random Vectors. (arXiv:2401.13429v1 [cs.IT])

    [http://arxiv.org/abs/2401.13429](http://arxiv.org/abs/2401.13429)

    本文研究了判断两个标准正态随机向量是否相关的问题，提出了一种新的方法来评估似然比的二阶矩，并发现了与整数分割函数之间的联系。

    

    在本文中，我们研究了判断两个标准正态随机向量$\mathsf{X}\in\mathbb{R}^{n}$和$\mathsf{Y}\in\mathbb{R}^{n}$是否相关的问题。这被表述为一个假设检验问题，在零假设下，这些向量是统计独立的，而在备择假设下，$\mathsf{X}$和随机均匀置换的$\mathsf{Y}$是具有相关系数$\rho$的。我们分析了信息论上不可能和可能的最优测试阈值，作为$n$和$\rho$的函数。为了得出我们的信息论下界，我们开发了一种利用正交多项式展开来评估似然比的二阶矩的新技术，该技术揭示了与整数分割函数之间的一个令人惊讶的联系。我们还研究了上述设置的多维泛化，其中我们观察到两个数据库/矩阵，而不是两个向量。

    In this paper, we investigate the problem of deciding whether two standard normal random vectors $\mathsf{X}\in\mathbb{R}^{n}$ and $\mathsf{Y}\in\mathbb{R}^{n}$ are correlated or not. This is formulated as a hypothesis testing problem, where under the null hypothesis, these vectors are statistically independent, while under the alternative, $\mathsf{X}$ and a randomly and uniformly permuted version of $\mathsf{Y}$, are correlated with correlation $\rho$. We analyze the thresholds at which optimal testing is information-theoretically impossible and possible, as a function of $n$ and $\rho$. To derive our information-theoretic lower bounds, we develop a novel technique for evaluating the second moment of the likelihood ratio using an orthogonal polynomials expansion, which among other things, reveals a surprising connection to integer partition functions. We also study a multi-dimensional generalization of the above setting, where rather than two vectors we observe two databases/matrices
    
[^2]: 随机响应私有置信集的非参数扩展

    Nonparametric extensions of randomized response for private confidence sets. (arXiv:2202.08728v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.08728](http://arxiv.org/abs/2202.08728)

    本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。

    

    本文提出了一种在局部差分隐私（LDP）约束下执行非参数、非渐进统计推断的方法，用于计算具有均值$\mu^\star$的有界观测$(X_1,\dots,X_n)$的置信区间（CI）和时间均匀置信序列（CS），当只有访问私有数据$(Z_1,\dots,Z_n)$时。为了实现这一点，我们引入了一个非参数的、顺序交互的 Warner 的著名“随机响应”机制的推广，为任意有界随机变量满足 LDP，并提供 CIs 和 CSs，用于访问所得私有化的观测值的均值。例如，我们的结果在固定时间和时间均匀区域都产生了 Hoeffding 不等式的私有模拟。我们将这些 Hoeffding  类型的 CSs 扩展到捕获时间变化（非平稳）的均值，最后说明了如何利用这些方法进行实证。

    This work derives methods for performing nonparametric, nonasymptotic statistical inference for population means under the constraint of local differential privacy (LDP). Given bounded observations $(X_1, \dots, X_n)$ with mean $\mu^\star$ that are privatized into $(Z_1, \dots, Z_n)$, we present confidence intervals (CI) and time-uniform confidence sequences (CS) for $\mu^\star$ when only given access to the privatized data. To achieve this, we introduce a nonparametric and sequentially interactive generalization of Warner's famous ``randomized response'' mechanism, satisfying LDP for arbitrary bounded random variables, and then provide CIs and CSs for their means given access to the resulting privatized observations. For example, our results yield private analogues of Hoeffding's inequality in both fixed-time and time-uniform regimes. We extend these Hoeffding-type CSs to capture time-varying (non-stationary) means, and conclude by illustrating how these methods can be used to conduct
    

