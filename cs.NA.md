# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence analysis of online algorithms for vector-valued kernel regression.](http://arxiv.org/abs/2309.07779) | 本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。 |

# 详细

[^1]: 在向量值内核回归的在线算法的收敛分析

    Convergence analysis of online algorithms for vector-valued kernel regression. (arXiv:2309.07779v1 [stat.ML])

    [http://arxiv.org/abs/2309.07779](http://arxiv.org/abs/2309.07779)

    本文考虑了在线学习算法在向量值内核回归问题中的收敛性能，证明了在RKHS范数中的期望平方误差可以被一个特定公式所限制。

    

    我们考虑使用适当的再生核希尔伯特空间（RKHS）作为先验，通过在线学习算法从噪声向量值数据中逼近回归函数的问题。在在线算法中，独立同分布的样本通过随机过程逐个可用，并依次处理以构建对回归函数的近似。我们关注这种在线逼近算法的渐近性能，并证明了在RKHS范数中的期望平方误差可以被$C^2(m+1)^{-s/(2+s)}$绑定，其中$m$为当下处理的数据数量，参数$0<s\leq 1$表示对回归函数的额外光滑性假设，常数$C$取决于输入噪声的方差、回归函数的光滑性以及算法的其他参数。

    We consider the problem of approximating the regression function from noisy vector-valued data by an online learning algorithm using an appropriate reproducing kernel Hilbert space (RKHS) as prior. In an online algorithm, i.i.d. samples become available one by one by a random process and are successively processed to build approximations to the regression function. We are interested in the asymptotic performance of such online approximation algorithms and show that the expected squared error in the RKHS norm can be bounded by $C^2 (m+1)^{-s/(2+s)}$, where $m$ is the current number of processed data, the parameter $0<s\leq 1$ expresses an additional smoothness assumption on the regression function and the constant $C$ depends on the variance of the input noise, the smoothness of the regression function and further parameters of the algorithm.
    

