# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Auditing Fairness by Betting.](http://arxiv.org/abs/2305.17570) | 本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。 |

# 详细

[^1]: 通过赌博进行公平性审计

    Auditing Fairness by Betting. (arXiv:2305.17570v1 [stat.ML])

    [http://arxiv.org/abs/2305.17570](http://arxiv.org/abs/2305.17570)

    本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。

    

    我们提供了实用、高效、非参数方法，用于审计已部署的分类和回归模型的公平性。相比之前依赖于固定样本量的方法，我们的方法是序贯的，并允许对不断产生的数据进行连续的监控，因此非常适用于跟踪现实世界系统的公平性。我们也允许数据通过概率策略进行收集，而不是从人口中均匀采样。这使得审计可以在为其他目的收集的数据上进行。此外，该策略可以随时间改变，并且不同的子人群可以使用不同的策略。最后，我们的方法可以处理因模型变更或基础人群变更导致的分布漂移。我们的方法基于最近关于 anytime-valid 推断和博弈统计学的进展，尤其是"通过赌博进行测试"框架。这些联系确保了我们的方法具有可解释性、快速和提供统计保证。

    We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics-the "testing by betting" framework in particular. These connections ensure that our methods are interpretable, fast, 
    

