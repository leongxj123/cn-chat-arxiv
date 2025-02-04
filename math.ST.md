# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Horseshoe Gaussian Processes](https://arxiv.org/abs/2403.01737) | 深马蹄高斯过程Deep-HGP是一种简单的先验，采用深高斯过程并允许数据驱动选择关键长度尺度参数，对于非参数回归表现出良好的性能，实现了对未知真实回归曲线的优化回复，具有自适应的收敛速率。 |
| [^2] | [Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data.](http://arxiv.org/abs/2310.12140) | 本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性 |

# 详细

[^1]: 深马蹄高斯过程

    Deep Horseshoe Gaussian Processes

    [https://arxiv.org/abs/2403.01737](https://arxiv.org/abs/2403.01737)

    深马蹄高斯过程Deep-HGP是一种简单的先验，采用深高斯过程并允许数据驱动选择关键长度尺度参数，对于非参数回归表现出良好的性能，实现了对未知真实回归曲线的优化回复，具有自适应的收敛速率。

    

    最近提出深高斯过程作为一种自然对象，类似于深度神经网络，可能拟合现代数据样本中存在的复杂特征，如组合结构。采用贝叶斯非参数方法，自然地利用深高斯过程作为先验分布，并将相应的后验分布用于统计推断。我们介绍了深马蹄高斯过程Deep-HGP，这是一种基于带有平方指数核的深高斯过程的新简单先验，特别是使得可以对关键长度尺度参数进行数据驱动选择。对于随机设计的非参数回归，我们展示了相应的调节后验分布以一种自适应方式，最优地在二次损失的意义下恢复未知的真回归曲线，最多只有一个对数因子。收敛速率同时对回归的平滑度和设计维度自适应。

    arXiv:2403.01737v1 Announce Type: cross  Abstract: Deep Gaussian processes have recently been proposed as natural objects to fit, similarly to deep neural networks, possibly complex features present in modern data samples, such as compositional structures. Adopting a Bayesian nonparametric approach, it is natural to use deep Gaussian processes as prior distributions, and use the corresponding posterior distributions for statistical inference. We introduce the deep Horseshoe Gaussian process Deep-HGP, a new simple prior based on deep Gaussian processes with a squared-exponential kernel, that in particular enables data-driven choices of the key lengthscale parameters. For nonparametric regression with random design, we show that the associated tempered posterior distribution recovers the unknown true regression curve optimally in terms of quadratic loss, up to a logarithmic factor, in an adaptive way. The convergence rates are simultaneously adaptive to both the smoothness of the regress
    
[^2]: 在线估计与滚动验证：适应性非参数估计与数据流

    Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data. (arXiv:2310.12140v1 [math.ST])

    [http://arxiv.org/abs/2310.12140](http://arxiv.org/abs/2310.12140)

    本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性

    

    由于其高效计算和竞争性的泛化能力，在线非参数估计器越来越受欢迎。一个重要的例子是随机梯度下降的变体。这些算法通常一次只取一个样本点，并立即更新感兴趣的参数估计。在这项工作中，我们考虑了这些在线算法的模型选择和超参数调整。我们提出了一种加权滚动验证过程，一种在线的留一交叉验证变体，对于许多典型的随机梯度下降估计器来说，额外的计算成本最小。类似于批量交叉验证，它可以提升基本估计器的自适应收敛速度。我们的理论分析很简单，主要依赖于一些一般的统计稳定性假设。模拟研究强调了滚动验证中发散权重在实践中的重要性，并证明了即使只有一个很小的偏差，它的敏感性也很高

    Online nonparametric estimators are gaining popularity due to their efficient computation and competitive generalization abilities. An important example includes variants of stochastic gradient descent. These algorithms often take one sample point at a time and instantly update the parameter estimate of interest. In this work we consider model selection and hyperparameter tuning for such online algorithms. We propose a weighted rolling-validation procedure, an online variant of leave-one-out cross-validation, that costs minimal extra computation for many typical stochastic gradient descent estimators. Similar to batch cross-validation, it can boost base estimators to achieve a better, adaptive convergence rate. Our theoretical analysis is straightforward, relying mainly on some general statistical stability assumptions. The simulation study underscores the significance of diverging weights in rolling validation in practice and demonstrates its sensitivity even when there is only a slim
    

