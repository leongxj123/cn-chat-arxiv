# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Communication-Efficient Distributed Estimation and Inference for Cox's Model.](http://arxiv.org/abs/2302.12111) | 我们提出了一种高效的分布式算法，用于在高维稀疏Cox比例风险模型中估计和推断，通过引入一种新的去偏差方法，我们可以产生渐近有效的分布式置信区间，并提供了有效的分布式假设检验。 |

# 详细

[^1]: 面向Cox模型的高效通信式分布式估计和推断

    Communication-Efficient Distributed Estimation and Inference for Cox's Model. (arXiv:2302.12111v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.12111](http://arxiv.org/abs/2302.12111)

    我们提出了一种高效的分布式算法，用于在高维稀疏Cox比例风险模型中估计和推断，通过引入一种新的去偏差方法，我们可以产生渐近有效的分布式置信区间，并提供了有效的分布式假设检验。

    

    针对因隐私和所有权问题无法共享个体数据的多中心生物医学研究，我们开发了高维稀疏Cox比例风险模型的通信高效迭代分布式算法用于估计和推断。我们证明了即使进行了相对较少的迭代，我们的估计值在非常温和的条件下可以达到与理想全样本估计值相同的收敛速度。为了构建高维危险回归系数的线性组合的置信区间，我们引入了一种新的去偏差方法，建立了中心极限定理，并提供了一致的方差估计，可以产生渐近有效的分布式置信区间。此外，我们提供了基于装饰分数检验的任意坐标元素的有效和强大的分布式假设检验。我们还允许时间依赖协变量以及被审查的生存时间。在多种数据集上进行了广泛的数字实验，证明了算法的有效性和效率。

    Motivated by multi-center biomedical studies that cannot share individual data due to privacy and ownership concerns, we develop communication-efficient iterative distributed algorithms for estimation and inference in the high-dimensional sparse Cox proportional hazards model. We demonstrate that our estimator, even with a relatively small number of iterations, achieves the same convergence rate as the ideal full-sample estimator under very mild conditions. To construct confidence intervals for linear combinations of high-dimensional hazard regression coefficients, we introduce a novel debiased method, establish central limit theorems, and provide consistent variance estimators that yield asymptotically valid distributed confidence intervals. In addition, we provide valid and powerful distributed hypothesis tests for any coordinate element based on a decorrelated score test. We allow time-dependent covariates as well as censored survival times. Extensive numerical experiments on both s
    

