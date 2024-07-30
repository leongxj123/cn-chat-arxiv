# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Wasserstein perspective of Vanilla GANs](https://arxiv.org/abs/2403.15312) | 将普通GANs与水斯坦距离联系起来，扩展现有水斯坦GANs结果到普通GANs，获得了普通GANs的神谕不等式。 |
| [^2] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |
| [^3] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |
| [^4] | [Optimal Pre-Analysis Plans: Statistical Decisions Subject to Implementability.](http://arxiv.org/abs/2208.09638) | 本研究提出了一个委托-代理模型来解决先分析计划的设计问题，通过实施具有可行性的统计决策规则，发现先分析计划在实施中起着重要作用，特别是对于假设检验来说，最优拒绝规则需要预先注册有效的检验并对未报告的数据做最坏情况假设。 |
| [^5] | [Multitask Learning and Bandits via Robust Statistics.](http://arxiv.org/abs/2112.14233) | 本研究探讨了多任务学习以及Bandits方法的健壮统计学实现，提出了一种新颖的两阶段多任务学习估计器，该估计器以一种样本高效的方式利用共享全局参数和稀疏实例特定术语的结构。 |

# 详细

[^1]: 水斯坦视角下的普通 GANs

    A Wasserstein perspective of Vanilla GANs

    [https://arxiv.org/abs/2403.15312](https://arxiv.org/abs/2403.15312)

    将普通GANs与水斯坦距离联系起来，扩展现有水斯坦GANs结果到普通GANs，获得了普通GANs的神谕不等式。

    

    生成对抗网络(GANs)的实证成功引起了对理论研究日益增长的兴趣。统计文献主要集中在水斯坦GANs及其扩展上，特别是允许具有良好的降维特性。对于普通GANs，即原始优化问题，统计结果仍然相当有限，需要假设平滑激活函数和潜空间与周围空间的维度相等。为了弥合这一差距，我们将普通GANs与水斯坦距离联系起来。通过这样做，现有的水斯坦GANs结果可以扩展到普通GANs。特别是，在水斯坦距离中获得了普通GANs的神谕不等式。这个神谕不等式的假设旨在由实践中常用的网络架构满足，如前馈ReLU网络。

    arXiv:2403.15312v1 Announce Type: cross  Abstract: The empirical success of Generative Adversarial Networks (GANs) caused an increasing interest in theoretical research. The statistical literature is mainly focused on Wasserstein GANs and generalizations thereof, which especially allow for good dimension reduction properties. Statistical results for Vanilla GANs, the original optimization problem, are still rather limited and require assumptions such as smooth activation functions and equal dimensions of the latent space and the ambient space. To bridge this gap, we draw a connection from Vanilla GANs to the Wasserstein distance. By doing so, existing results for Wasserstein GANs can be extended to Vanilla GANs. In particular, we obtain an oracle inequality for Vanilla GANs in Wasserstein distance. The assumptions of this oracle inequality are designed to be satisfied by network architectures commonly used in practice, such as feedforward ReLU networks. By providing a quantitative resu
    
[^2]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    
[^3]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    
[^4]: 最优先分析计划：受限于可实施性的统计决策

    Optimal Pre-Analysis Plans: Statistical Decisions Subject to Implementability. (arXiv:2208.09638v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2208.09638](http://arxiv.org/abs/2208.09638)

    本研究提出了一个委托-代理模型来解决先分析计划的设计问题，通过实施具有可行性的统计决策规则，发现先分析计划在实施中起着重要作用，特别是对于假设检验来说，最优拒绝规则需要预先注册有效的检验并对未报告的数据做最坏情况假设。

    

    什么是先分析计划的目的，应该如何设计？我们提出了一个委托-代理模型，其中决策者依赖于分析师选择性但真实的报告。分析师具有数据访问权，且目标不一致。在这个模型中，实施统计决策规则（检验、估计量）需要一个激励兼容机制。我们首先描述了哪些决策规则可以被实施。然后描述了受限于可实施性的最优统计决策规则。我们表明实施需要先分析计划。重点放在假设检验上，我们表明最优拒绝规则预先注册了一个对于所有数据报告的有效检验，并对未报告的数据做最坏情况假设。最优检验可以通过线性规划问题的解来找到。

    What is the purpose of pre-analysis plans, and how should they be designed? We propose a principal-agent model where a decision-maker relies on selective but truthful reports by an analyst. The analyst has data access, and non-aligned objectives. In this model, the implementation of statistical decision rules (tests, estimators) requires an incentive-compatible mechanism. We first characterize which decision rules can be implemented. We then characterize optimal statistical decision rules subject to implementability. We show that implementation requires pre-analysis plans. Focussing specifically on hypothesis tests, we show that optimal rejection rules pre-register a valid test for the case when all data is reported, and make worst-case assumptions about unreported data. Optimal tests can be found as a solution to a linear-programming problem.
    
[^5]: 多任务学习和Bandits通过健壮统计学

    Multitask Learning and Bandits via Robust Statistics. (arXiv:2112.14233v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2112.14233](http://arxiv.org/abs/2112.14233)

    本研究探讨了多任务学习以及Bandits方法的健壮统计学实现，提出了一种新颖的两阶段多任务学习估计器，该估计器以一种样本高效的方式利用共享全局参数和稀疏实例特定术语的结构。

    

    决策者经常同时面对许多相关但异质的学习问题。在此工作中，我们研究了一种自然的设置，其中每个学习实例中的未知参数可以分解为共享全局参数加上稀疏的实例特定术语。我们提出了一种新颖的两阶段多任务学习估计器，以一种样本高效的方式利用这种结构，使用健壮统计学（在相似实例上学习）和LASSO回归（去偏差结果）的独特组合。我们的估计器提供了改进的样本复杂度界限。

    Decision-makers often simultaneously face many related but heterogeneous learning problems. For instance, a large retailer may wish to learn product demand at different stores to solve pricing or inventory problems, making it desirable to learn jointly for stores serving similar customers; alternatively, a hospital network may wish to learn patient risk at different providers to allocate personalized interventions, making it desirable to learn jointly for hospitals serving similar patient populations. Motivated by real datasets, we study a natural setting where the unknown parameter in each learning instance can be decomposed into a shared global parameter plus a sparse instance-specific term. We propose a novel two-stage multitask learning estimator that exploits this structure in a sample-efficient way, using a unique combination of robust statistics (to learn across similar instances) and LASSO regression (to debias the results). Our estimator yields improved sample complexity bound
    

