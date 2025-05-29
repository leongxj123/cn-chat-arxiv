# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal convex $M$-estimation via score matching](https://arxiv.org/abs/2403.16688) | 该论文提出了一种通过得分匹配实现最佳凸$M$-估计的方法，在线性回归中能够达到最佳的渐近方差，并且在计算上高效，证明具有所有凸$M$-估计中最小的渐近协方差。 |
| [^2] | [Regularizing Discrimination in Optimal Policy Learning with Distributional Targets](https://arxiv.org/abs/2401.17909) | 为了解决优化策略学习中的歧视问题，研究者提出了一个框架，允许决策者通过惩罚来防止在特定人群中的不公平结果分布，该框架对目标函数和歧视度量具有很大的灵活性，通过数据驱动的参数调整，可以在实践中具备遗憾和一致性保证。 |

# 详细

[^1]: 通过得分匹配实现最佳凸$M$-估计

    Optimal convex $M$-estimation via score matching

    [https://arxiv.org/abs/2403.16688](https://arxiv.org/abs/2403.16688)

    该论文提出了一种通过得分匹配实现最佳凸$M$-估计的方法，在线性回归中能够达到最佳的渐近方差，并且在计算上高效，证明具有所有凸$M$-估计中最小的渐近协方差。

    

    在线性回归的背景下，我们构建了一个数据驱动的凸损失函数，通过该函数进行经验风险最小化可以在回归系数的下游估计中实现最佳的渐近方差。我们的半参数方法旨在最佳逼近噪声分布对数密度的导数。在总体层面上，这个拟合过程是对得分匹配的非参数拓展，对应于根据Fisher散度进行噪声分布的对数凹映射。该过程在计算上是高效的，我们证明我们的程序达到了所有凸$M$-估计中最小的渐近协方差。作为非对数凹设置的一个例子，对于柯西误差，最佳凸损失函数类似于Huber函数，并且我们的过程相对于oracle最大似然估计器实现了大于0.87的渐近效率。

    arXiv:2403.16688v1 Announce Type: cross  Abstract: In the context of linear regression, we construct a data-driven convex loss function with respect to which empirical risk minimisation yields optimal asymptotic variance in the downstream estimation of the regression coefficients. Our semiparametric approach targets the best decreasing approximation of the derivative of the log-density of the noise distribution. At the population level, this fitting process is a nonparametric extension of score matching, corresponding to a log-concave projection of the noise distribution with respect to the Fisher divergence. The procedure is computationally efficient, and we prove that our procedure attains the minimal asymptotic covariance among all convex $M$-estimators. As an example of a non-log-concave setting, for Cauchy errors, the optimal convex loss function is Huber-like, and our procedure yields an asymptotic efficiency greater than 0.87 relative to the oracle maximum likelihood estimator o
    
[^2]: 优化策略学习中正则化歧视问题的研究

    Regularizing Discrimination in Optimal Policy Learning with Distributional Targets

    [https://arxiv.org/abs/2401.17909](https://arxiv.org/abs/2401.17909)

    为了解决优化策略学习中的歧视问题，研究者提出了一个框架，允许决策者通过惩罚来防止在特定人群中的不公平结果分布，该框架对目标函数和歧视度量具有很大的灵活性，通过数据驱动的参数调整，可以在实践中具备遗憾和一致性保证。

    

    决策者通常通过训练数据学习治疗的相对效果，并选择一个实施机制，该机制根据某个目标函数预测了“最优”结果分布。然而，一个意识到歧视问题的决策者可能不满意以严重歧视人群子组的代价来实现该优化，即在子组中的结果分布明显偏离整体最优结果分布。我们研究了一个框架，允许决策者惩罚这种偏差，并可以使用各种目标函数和歧视度量。我们对具有数据驱动调参的经验成功策略建立了遗憾和一致性保证，并提供了数值结果。此外，我们还对两个实证场景进行了简要说明。

    A decision maker typically (i) incorporates training data to learn about the relative effectiveness of the treatments, and (ii) chooses an implementation mechanism that implies an "optimal" predicted outcome distribution according to some target functional. Nevertheless, a discrimination-aware decision maker may not be satisfied achieving said optimality at the cost of heavily discriminating against subgroups of the population, in the sense that the outcome distribution in a subgroup deviates strongly from the overall optimal outcome distribution. We study a framework that allows the decision maker to penalize for such deviations, while allowing for a wide range of target functionals and discrimination measures to be employed. We establish regret and consistency guarantees for empirical success policies with data-driven tuning parameters, and provide numerical results. Furthermore, we briefly illustrate the methods in two empirical settings.
    

