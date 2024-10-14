# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric consistency for maximum likelihood estimation and clustering based on mixtures of elliptically-symmetric distributions](https://arxiv.org/abs/2311.06108) | 展示了椭圆对称分布混合的最大似然估计的一致性，为基于非参数分布的聚类提供了理论依据。 |
| [^2] | [Unlocking Unlabeled Data: Ensemble Learning with the Hui- Walter Paradigm for Performance Estimation in Online and Static Settings.](http://arxiv.org/abs/2401.09376) | 该论文通过适应Hui-Walter范式，将传统应用于流行病学和医学的方法引入机器学习领域，解决了训练和评估时无法获得标签数据的问题。通过将数据划分为潜在类别，并在多个测试中独立训练模型，能够在没有真实值的情况下估计关键性能指标，并在处理在线数据时提供了新的可能性。 |
| [^3] | [Certified Multi-Fidelity Zeroth-Order Optimization.](http://arxiv.org/abs/2308.00978) | 本文研究了认证的多流程零阶优化问题，提出了MFDOO算法的认证变体，并证明了其具有近似最优的代价复杂度。同时，还考虑了有噪声评估的特殊情况。 |

# 详细

[^1]: 基于椭圆对称分布混合的最大似然估计和聚类的非参数一致性

    Nonparametric consistency for maximum likelihood estimation and clustering based on mixtures of elliptically-symmetric distributions

    [https://arxiv.org/abs/2311.06108](https://arxiv.org/abs/2311.06108)

    展示了椭圆对称分布混合的最大似然估计的一致性，为基于非参数分布的聚类提供了理论依据。

    

    该论文展示了椭圆对称分布混合的最大似然估计器对其总体版本的一致性，其中潜在分布P是非参数的，并不一定属于估计器所基于的混合类别。当P是足够分离但非参数的分布混合时，表明了估计器的总体版本的组分对应于P的良好分离组分。这为在P具有良好分离子总体的情况下使用这样的估计器进行聚类分析提供了一些理论上的理据，即使这些子总体与混合模型所假设的不同。

    arXiv:2311.06108v2 Announce Type: replace-cross  Abstract: The consistency of the maximum likelihood estimator for mixtures of elliptically-symmetric distributions for estimating its population version is shown, where the underlying distribution $P$ is nonparametric and does not necessarily belong to the class of mixtures on which the estimator is based. In a situation where $P$ is a mixture of well enough separated but nonparametric distributions it is shown that the components of the population version of the estimator correspond to the well separated components of $P$. This provides some theoretical justification for the use of such estimators for cluster analysis in case that $P$ has well separated subpopulations even if these subpopulations differ from what the mixture model assumes.
    
[^2]: 解锁无标签数据: Hui-Walter范式在在线和静态环境中的性能评估中的集成学习

    Unlocking Unlabeled Data: Ensemble Learning with the Hui- Walter Paradigm for Performance Estimation in Online and Static Settings. (arXiv:2401.09376v1 [cs.LG])

    [http://arxiv.org/abs/2401.09376](http://arxiv.org/abs/2401.09376)

    该论文通过适应Hui-Walter范式，将传统应用于流行病学和医学的方法引入机器学习领域，解决了训练和评估时无法获得标签数据的问题。通过将数据划分为潜在类别，并在多个测试中独立训练模型，能够在没有真实值的情况下估计关键性能指标，并在处理在线数据时提供了新的可能性。

    

    在机器学习和统计建模领域，从业人员常常在可评估和训练的假设下工作，即可访问的、静态的、带有标签的数据。然而，这个假设往往偏离了现实，其中的数据可能是私有的、加密的、难以测量的或者没有标签。本文通过将传统应用于流行病学和医学的Hui-Walter范式调整到机器学习领域来弥合这个差距。这种方法使我们能够在没有真实值可用的情况下估计关键性能指标，如假阳性率、假阴性率和先验概率。我们进一步扩展了这种范式来处理在线数据，开辟了动态数据环境的新可能性。我们的方法涉及将数据划分为潜在类别，以模拟多个数据群体（如果没有自然群体可用），并独立训练模型来复制多次测试。通过在不同数据子集之间交叉制表，我们能够比较二元结果。

    In the realm of machine learning and statistical modeling, practitioners often work under the assumption of accessible, static, labeled data for evaluation and training. However, this assumption often deviates from reality where data may be private, encrypted, difficult- to-measure, or unlabeled. In this paper, we bridge this gap by adapting the Hui-Walter paradigm, a method traditionally applied in epidemiology and medicine, to the field of machine learning. This approach enables us to estimate key performance metrics such as false positive rate, false negative rate, and priors in scenarios where no ground truth is available. We further extend this paradigm for handling online data, opening up new possibilities for dynamic data environments. Our methodology involves partitioning data into latent classes to simulate multiple data populations (if natural populations are unavailable) and independently training models to replicate multiple tests. By cross-tabulating binary outcomes across
    
[^3]: 认证的多流程零阶优化

    Certified Multi-Fidelity Zeroth-Order Optimization. (arXiv:2308.00978v1 [cs.LG])

    [http://arxiv.org/abs/2308.00978](http://arxiv.org/abs/2308.00978)

    本文研究了认证的多流程零阶优化问题，提出了MFDOO算法的认证变体，并证明了其具有近似最优的代价复杂度。同时，还考虑了有噪声评估的特殊情况。

    

    我们考虑多流程零阶优化的问题，在这个问题中，可以在不同的近似水平（代价不同）上评估函数$f$，目标是以尽可能低的代价优化$f$。在本文中，我们研究了\emph{认证}算法，它们额外要求输出一个对优化误差的数据驱动上界。我们首先以算法和评估环境之间的极小极大博弈形式来形式化问题。然后，我们提出了MFDOO算法的认证变体，并推导出其在任意Lipschitz函数$f$上的代价复杂度上界。我们还证明了一个依赖于$f$的下界，表明该算法具有近似最优的代价复杂度。最后，我们通过直接示例解决了有噪声（随机）评估的特殊情况。

    We consider the problem of multi-fidelity zeroth-order optimization, where one can evaluate a function $f$ at various approximation levels (of varying costs), and the goal is to optimize $f$ with the cheapest evaluations possible. In this paper, we study \emph{certified} algorithms, which are additionally required to output a data-driven upper bound on the optimization error. We first formalize the problem in terms of a min-max game between an algorithm and an evaluation environment. We then propose a certified variant of the MFDOO algorithm and derive a bound on its cost complexity for any Lipschitz function $f$. We also prove an $f$-dependent lower bound showing that this algorithm has a near-optimal cost complexity. We close the paper by addressing the special case of noisy (stochastic) evaluations as a direct example.
    

