# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bias-Aware Inference in Regularized Regression Models.](http://arxiv.org/abs/2012.14823) | 本文研究了在控制系数幅值约束下的标量回归系数推论问题。我们提出了一个基于正则化倾向得分回归的估计器类，能够在最坏情况下权衡偏差和方差。通过导出偏差感知的置信区间，我们展示了估计器可能偏差的考虑。在均方误差和置信区间长度方面，这些估计器在同方差高斯误差下在有限样本中接近最优。我们还证明了在未知和可能异方差的误差分布下置信区间的渐近有效性，并在高维情况下导出了新的最优收敛速率，允许回归变量数量比观测数增长更快。大量的模拟实验和实证应用证明了我们方法的性能。 |

# 详细

[^1]: 基于正则化回归模型的偏差感知推论

    Bias-Aware Inference in Regularized Regression Models. (arXiv:2012.14823v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2012.14823](http://arxiv.org/abs/2012.14823)

    本文研究了在控制系数幅值约束下的标量回归系数推论问题。我们提出了一个基于正则化倾向得分回归的估计器类，能够在最坏情况下权衡偏差和方差。通过导出偏差感知的置信区间，我们展示了估计器可能偏差的考虑。在均方误差和置信区间长度方面，这些估计器在同方差高斯误差下在有限样本中接近最优。我们还证明了在未知和可能异方差的误差分布下置信区间的渐近有效性，并在高维情况下导出了新的最优收敛速率，允许回归变量数量比观测数增长更快。大量的模拟实验和实证应用证明了我们方法的性能。

    

    本文考虑在控制系数幅值约束下对标量回归系数进行推论。我们展示了一类基于正则化倾向得分回归的估计器能够精确地解决最坏情况下偏差和方差之间的权衡。我们基于这些估计器导出了偏差感知的置信区间(CIs)，它们考虑了估计器的可能偏差。在均方误差(MSE)和置信区间长度方面，对于同方差的高斯误差，这些估计器和CIs在有限样本中接近最优。我们还提供了在未知和可能异方差的误差分布下置信区间的渐近有效性条件，并导出了高维渐近下允许回归变量数量比观测数增长更快的新的最优收敛速率。大量的模拟实验和实证应用验证了我们方法的性能。

    We consider inference on a scalar regression coefficient under a constraint on the magnitude of the control coefficients. A class of estimators based on a regularized propensity score regression is shown to exactly solve a tradeoff between worst-case bias and variance. We derive confidence intervals (CIs) based on these estimators that are bias-aware: they account for the possible bias of the estimator. Under homoskedastic Gaussian errors, these estimators and CIs are near-optimal in finite samples for MSE and CI length. We also provide conditions for asymptotic validity of the CI with unknown and possibly heteroskedastic error distribution, and derive novel optimal rates of convergence under high-dimensional asymptotics that allow the number of regressors to increase more quickly than the number of observations. Extensive simulations and an empirical application illustrate the performance of our methods.
    

