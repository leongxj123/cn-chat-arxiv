# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frequentist Guarantees of Distributed (Non)-Bayesian Inference](https://arxiv.org/abs/2311.08214) | 本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。 |
| [^2] | [Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data.](http://arxiv.org/abs/2310.12140) | 本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性 |
| [^3] | [Statistical guarantees for stochastic Metropolis-Hastings.](http://arxiv.org/abs/2310.09335) | 该论文研究了针对随机Metropolis-Hastings算法的统计保证。通过引入简单的修正项，该方法可以避免计算成本上的损失，并通过分析非参数回归情景和深度神经网络回归的数值实例来证明了其在采样和可信区间方面的优势。 |

# 详细

[^1]: 分布式(非)贝叶斯推断的频率保证

    Frequentist Guarantees of Distributed (Non)-Bayesian Inference

    [https://arxiv.org/abs/2311.08214](https://arxiv.org/abs/2311.08214)

    本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。

    

    受分析大型分散数据集的需求推动，分布式贝叶斯推断已成为跨多个领域（包括统计学、电气工程和经济学）的关键研究领域。本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，如后验一致性、渐近正态性和后验收缩率。我们的结果表明，在通信图上的适当假设下，分布式贝叶斯推断保留了参数效率，同时在不确定性量化方面增强了鲁棒性。我们还通过研究设计和通信图的大小如何影响后验收缩率来探讨了统计效率和通信效率之间的权衡。此外，我们将我们的分析扩展到时变图，并将结果应用于指数f

    arXiv:2311.08214v2 Announce Type: replace-cross  Abstract: Motivated by the need to analyze large, decentralized datasets, distributed Bayesian inference has become a critical research area across multiple fields, including statistics, electrical engineering, and economics. This paper establishes Frequentist properties, such as posterior consistency, asymptotic normality, and posterior contraction rates, for the distributed (non-)Bayes Inference problem among agents connected via a communication network. Our results show that, under appropriate assumptions on the communication graph, distributed Bayesian inference retains parametric efficiency while enhancing robustness in uncertainty quantification. We also explore the trade-off between statistical efficiency and communication efficiency by examining how the design and size of the communication graph impact the posterior contraction rate. Furthermore, We extend our analysis to time-varying graphs and apply our results to exponential f
    
[^2]: 在线估计与滚动验证：适应性非参数估计与数据流

    Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data. (arXiv:2310.12140v1 [math.ST])

    [http://arxiv.org/abs/2310.12140](http://arxiv.org/abs/2310.12140)

    本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性

    

    由于其高效计算和竞争性的泛化能力，在线非参数估计器越来越受欢迎。一个重要的例子是随机梯度下降的变体。这些算法通常一次只取一个样本点，并立即更新感兴趣的参数估计。在这项工作中，我们考虑了这些在线算法的模型选择和超参数调整。我们提出了一种加权滚动验证过程，一种在线的留一交叉验证变体，对于许多典型的随机梯度下降估计器来说，额外的计算成本最小。类似于批量交叉验证，它可以提升基本估计器的自适应收敛速度。我们的理论分析很简单，主要依赖于一些一般的统计稳定性假设。模拟研究强调了滚动验证中发散权重在实践中的重要性，并证明了即使只有一个很小的偏差，它的敏感性也很高

    Online nonparametric estimators are gaining popularity due to their efficient computation and competitive generalization abilities. An important example includes variants of stochastic gradient descent. These algorithms often take one sample point at a time and instantly update the parameter estimate of interest. In this work we consider model selection and hyperparameter tuning for such online algorithms. We propose a weighted rolling-validation procedure, an online variant of leave-one-out cross-validation, that costs minimal extra computation for many typical stochastic gradient descent estimators. Similar to batch cross-validation, it can boost base estimators to achieve a better, adaptive convergence rate. Our theoretical analysis is straightforward, relying mainly on some general statistical stability assumptions. The simulation study underscores the significance of diverging weights in rolling validation in practice and demonstrates its sensitivity even when there is only a slim
    
[^3]: 针对随机Metropolis-Hastings算法的统计保证

    Statistical guarantees for stochastic Metropolis-Hastings. (arXiv:2310.09335v1 [stat.ML])

    [http://arxiv.org/abs/2310.09335](http://arxiv.org/abs/2310.09335)

    该论文研究了针对随机Metropolis-Hastings算法的统计保证。通过引入简单的修正项，该方法可以避免计算成本上的损失，并通过分析非参数回归情景和深度神经网络回归的数值实例来证明了其在采样和可信区间方面的优势。

    

    Metropolis-Hastings步骤被广泛应用于基于梯度的马尔可夫链蒙特卡洛方法中的不确定性量化中。通过对批次计算接受概率，随机Metropolis-Hastings步骤节省了计算成本，但降低了有效样本量。我们展示了通过简单的修正项可以避免这个障碍。我们研究了如果在非参数回归设置中应用改进的随机Metropolis-Hastings方法从Gibbs后验分布中采样，则链的结果稳态分布的统计属性。针对深度神经网络回归，我们证明了PAC-Bayes预言不等式，它提供了最优的收缩速率，并分析了结果可信区间的直径和高置信概率。通过在高维参数空间中的数值实例，我们说明了随机Metropolis-Hastings算法的可信区间和收缩速率确实表现出类似的行为。

    A Metropolis-Hastings step is widely used for gradient-based Markov chain Monte Carlo methods in uncertainty quantification. By calculating acceptance probabilities on batches, a stochastic Metropolis-Hastings step saves computational costs, but reduces the effective sample size. We show that this obstacle can be avoided by a simple correction term. We study statistical properties of the resulting stationary distribution of the chain if the corrected stochastic Metropolis-Hastings approach is applied to sample from a Gibbs posterior distribution in a nonparametric regression setting. Focusing on deep neural network regression, we prove a PAC-Bayes oracle inequality which yields optimal contraction rates and we analyze the diameter and show high coverage probability of the resulting credible sets. With a numerical example in a high-dimensional parameter space, we illustrate that credible sets and contraction rates of the stochastic Metropolis-Hastings algorithm indeed behave similar to 
    

