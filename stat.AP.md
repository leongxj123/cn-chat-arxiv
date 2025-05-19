# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Local-Polynomial Estimation for Multivariate Regression Discontinuity Designs](https://arxiv.org/abs/2402.08941) | 在多元回归不连续设计中，我们提出了一种多元局部多项式估计方法，能够处理多元设计并捕捉边界处的异质治疗效应。 |
| [^2] | [Auditing Fairness by Betting.](http://arxiv.org/abs/2305.17570) | 本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。 |

# 详细

[^1]: 多元回归不连续设计的局部多项式估计方法

    Local-Polynomial Estimation for Multivariate Regression Discontinuity Designs

    [https://arxiv.org/abs/2402.08941](https://arxiv.org/abs/2402.08941)

    在多元回归不连续设计中，我们提出了一种多元局部多项式估计方法，能够处理多元设计并捕捉边界处的异质治疗效应。

    

    我们引入了一个多元局部线性估计器，用于处理多元回归不连续设计中的治疗分配问题。现有的方法使用从边界点到欧氏距离作为标量运行变量，因此多元设计被处理为单变量设计。然而，距离运行变量与渐近有效性的假设不相容。我们将多元设计作为多元处理。在这项研究中，我们开发了一种针对多元局部多项式估计器的新型渐近正常性。我们的估计器是渐近有效的，并能捕捉边界处的异质治疗效应。通过数值模拟，我们证明了我们估计器的有效性。我们在哥伦比亚奖学金研究中的实证说明揭示了治疗效应的更丰富的异质性（包括其不存在）。

    arXiv:2402.08941v1 Announce Type: new Abstract: We introduce a multivariate local-linear estimator for multivariate regression discontinuity designs in which treatment is assigned by crossing a boundary in the space of running variables. The dominant approach uses the Euclidean distance from a boundary point as the scalar running variable; hence, multivariate designs are handled as uni-variate designs. However, the distance running variable is incompatible with the assumption for asymptotic validity. We handle multivariate designs as multivariate. In this study, we develop a novel asymptotic normality for multivariate local-polynomial estimators. Our estimator is asymptotically valid and can capture heterogeneous treatment effects over the boundary. We demonstrate the effectiveness of our estimator through numerical simulations. Our empirical illustration of a Colombian scholarship study reveals a richer heterogeneity (including its absence) of the treatment effect that is hidden in th
    
[^2]: 通过赌博进行公平性审计

    Auditing Fairness by Betting. (arXiv:2305.17570v1 [stat.ML])

    [http://arxiv.org/abs/2305.17570](http://arxiv.org/abs/2305.17570)

    本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。

    

    我们提供了实用、高效、非参数方法，用于审计已部署的分类和回归模型的公平性。相比之前依赖于固定样本量的方法，我们的方法是序贯的，并允许对不断产生的数据进行连续的监控，因此非常适用于跟踪现实世界系统的公平性。我们也允许数据通过概率策略进行收集，而不是从人口中均匀采样。这使得审计可以在为其他目的收集的数据上进行。此外，该策略可以随时间改变，并且不同的子人群可以使用不同的策略。最后，我们的方法可以处理因模型变更或基础人群变更导致的分布漂移。我们的方法基于最近关于 anytime-valid 推断和博弈统计学的进展，尤其是"通过赌博进行测试"框架。这些联系确保了我们的方法具有可解释性、快速和提供统计保证。

    We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics-the "testing by betting" framework in particular. These connections ensure that our methods are interpretable, fast, 
    

