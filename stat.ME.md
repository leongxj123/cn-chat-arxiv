# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Monte Carlo inference for semiparametric Bayesian regression.](http://arxiv.org/abs/2306.05498) | 本文介绍了一种简单、通用和高效的半参数贝叶斯回归的蒙特卡洛推断策略，可用于联合后验一致性，即使经典的似然函数是难以处理或未知的。 |

# 详细

[^1]: 半参数贝叶斯回归的蒙特卡洛推断

    Monte Carlo inference for semiparametric Bayesian regression. (arXiv:2306.05498v1 [stat.ME])

    [http://arxiv.org/abs/2306.05498](http://arxiv.org/abs/2306.05498)

    本文介绍了一种简单、通用和高效的半参数贝叶斯回归的蒙特卡洛推断策略，可用于联合后验一致性，即使经典的似然函数是难以处理或未知的。

    

    数据转换对于参数回归模型的广泛适用性至关重要，但对于贝叶斯分析，联合推断转换和模型参数通常需要限制性参数转换或非参数表示，这对实现和理论分析来说计算效率低下且繁琐，限制了他们在实践中的可用性。本文介绍了一种简单、通用和高效的策略，直接通过将转换与独立变量和因变量的边缘分布相连的方式来定位未知转换和所有回归模型参数的后验分布，并通过贝叶斯非参数模型使用贝叶斯自举方法。关键是，这种方法在广泛的回归模型中都可以实现(1)联合后验一致性，包括多个模型错配情况，和(2)高效的蒙特卡罗算法，即使经典的似然函数是难以处理或未知的。

    Data transformations are essential for broad applicability of parametric regression models. However, for Bayesian analysis, joint inference of the transformation and model parameters typically involves restrictive parametric transformations or nonparametric representations that are computationally inefficient and cumbersome for implementation and theoretical analysis, which limits their usability in practice. This paper introduces a simple, general, and efficient strategy for joint posterior inference of an unknown transformation and all regression model parameters. The proposed approach directly targets the posterior distribution of the transformation by linking it with the marginal distributions of the independent and dependent variables, and then deploys a Bayesian nonparametric model via the Bayesian bootstrap. Crucially, this approach delivers (1) joint posterior consistency under general conditions, including multiple model misspecifications, and (2) efficient Monte Carlo (not Ma
    

