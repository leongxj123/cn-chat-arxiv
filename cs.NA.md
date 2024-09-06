# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multifidelity Covariance Estimation via Regression on the Manifold of Symmetric Positive Definite Matrices.](http://arxiv.org/abs/2307.12438) | 本论文介绍了一种在对称正定矩阵流形上进行回归求解的多保真度协方差估计器，通过构造满足正定性和可实际计算属性的马氏距离最小化。该估计器是最大似然估计器，并且能相对于其他方法显著减小估计误差。 |

# 详细

[^1]: 多保真度协方差估计：通过在对称正定矩阵流形上进行回归求解

    Multifidelity Covariance Estimation via Regression on the Manifold of Symmetric Positive Definite Matrices. (arXiv:2307.12438v2 [stat.CO] UPDATED)

    [http://arxiv.org/abs/2307.12438](http://arxiv.org/abs/2307.12438)

    本论文介绍了一种在对称正定矩阵流形上进行回归求解的多保真度协方差估计器，通过构造满足正定性和可实际计算属性的马氏距离最小化。该估计器是最大似然估计器，并且能相对于其他方法显著减小估计误差。

    

    我们介绍了一种多保真度协方差矩阵估计器，其构建为在对称正定矩阵流形上的回归问题的解。该估计器通过构造是正定的，并且其最小化的马氏距离具有可实际计算的属性。我们展示了我们的流形回归多保真度协方差估计器是在特定误差模型下的最大似然估计器。更广泛地说，我们展示了我们的黎曼回归框架包含了从控制变量构建的现有多保真度协方差估计器。我们通过数值示例证明，相对于单保真度和其他多保真度协方差估计器，我们的估计器可以显著减小估计误差的平方，减少一个数量级。此外，正定性的保持确保我们的估计器与下游任务兼容。

    We introduce a multifidelity estimator of covariance matrices formulated as the solution to a regression problem on the manifold of symmetric positive definite matrices. The estimator is positive definite by construction, and the Mahalanobis distance minimized to obtain it possesses properties which enable practical computation. We show that our manifold regression multifidelity (MRMF) covariance estimator is a maximum likelihood estimator under a certain error model on manifold tangent space. More broadly, we show that our Riemannian regression framework encompasses existing multifidelity covariance estimators constructed from control variates. We demonstrate via numerical examples that our estimator can provide significant decreases, up to one order of magnitude, in squared estimation error relative to both single-fidelity and other multifidelity covariance estimators. Furthermore, preservation of positive definiteness ensures that our estimator is compatible with downstream tasks, s
    

