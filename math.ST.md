# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels](https://arxiv.org/abs/2403.07735) | HSIC估计的极小化率对平移不变核的独立性度量具有重要意义 |
| [^2] | [Multivariate Tie-breaker Designs](https://arxiv.org/abs/2202.10030) | 该研究探讨了一种多元回归的打破平局设计，通过优化D-最优准则的治疗概率，实现了资源分配效率和统计效率的权衡。 |
| [^3] | [Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA.](http://arxiv.org/abs/2303.06198) | 本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。 |
| [^4] | [MARS via LASSO.](http://arxiv.org/abs/2111.11694) | 本文提出了一种自然lasso变体的MARS方法，通过减少对维度的依赖来获得收敛率，并与使用平滑性约束的非参数估计技术联系在一起。 |
| [^5] | [Optimal Scoring Rule Design under Partial Knowledge.](http://arxiv.org/abs/2107.07420) | 本文研究了在委托人对代理人的信号分布部分了解的情况下，最优打分规则的设计问题。作者提出了一个最大最小优化的框架，来最大化在代理人信号分布的集合中最坏情况下回报的增加。对于有限集合，提出了高效的算法；对于无限集合，提出了完全多项式时间逼近方案。 |

# 详细

[^1]: HSIC估计的极小化率对平移不变核

    The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels

    [https://arxiv.org/abs/2403.07735](https://arxiv.org/abs/2403.07735)

    HSIC估计的极小化率对平移不变核的独立性度量具有重要意义

    

    Kernel技术是数据科学和统计学中最有影响力的方法之一。在温和条件下，与核相关的再生核希尔伯特空间能够编码$M\ge 2$个随机变量的独立性。在核上依赖的最普遍的独立性度量可能是所谓的Hilbert-Schmidt独立性准则(HSIC; 在统计文献中也称为距离协方差)。尽管自近二十年前引入以来已经有各种现有的设计的HSIC估计量，HSIC可以被估计的速度的基本问题仍然是开放的。在这项工作中，我们证明了对于包含具有连续有界平移不变特征核的高斯Borel测度在$\mathbb R^d$上的HSIC估计的极小化最优速率是$\mathcal O\!\left(n^{-1/2}\right)$。具体地，我们的结果意味着许多方面在极小化意义上的最优性

    arXiv:2403.07735v1 Announce Type: cross  Abstract: Kernel techniques are among the most influential approaches in data science and statistics. Under mild conditions, the reproducing kernel Hilbert space associated to a kernel is capable of encoding the independence of $M\ge 2$ random variables. Probably the most widespread independence measure relying on kernels is the so-called Hilbert-Schmidt independence criterion (HSIC; also referred to as distance covariance in the statistics literature). Despite various existing HSIC estimators designed since its introduction close to two decades ago, the fundamental question of the rate at which HSIC can be estimated is still open. In this work, we prove that the minimax optimal rate of HSIC estimation on $\mathbb R^d$ for Borel measures containing the Gaussians with continuous bounded translation-invariant characteristic kernels is $\mathcal O\!\left(n^{-1/2}\right)$. Specifically, our result implies the optimality in the minimax sense of many 
    
[^2]: 多元化的打破平局设计

    Multivariate Tie-breaker Designs

    [https://arxiv.org/abs/2202.10030](https://arxiv.org/abs/2202.10030)

    该研究探讨了一种多元回归的打破平局设计，通过优化D-最优准则的治疗概率，实现了资源分配效率和统计效率的权衡。

    

    在打破平局设计（TBD）中，具有一定高值的运行变量的受试者接受某种（通常是理想的）治疗，低值的受试者不接受治疗，而中间的受试者被随机分配。 TBD介于回归断点设计（RDD）和随机对照试验（RCT）之间，通过允许在RDD的资源分配效率与RCT的统计效率之间进行权衡。 我们研究了一个模型，其中被治疗受试者的预期反应是一个多元回归，而对照受试者则是另一个。 对于给定的协变量，我们展示了如何使用凸优化来选择优化D-最优准则的治疗概率。 我们可以结合多种受经济和伦理考虑激发的约束条件。 在我们的模型中，对于治疗效应的D-最优性与整体回归的D-最优性重合，在没有经济约束的情况下，RCT即为最优选择。

    arXiv:2202.10030v4 Announce Type: replace-cross  Abstract: In a tie-breaker design (TBD), subjects with high values of a running variable are given some (usually desirable) treatment, subjects with low values are not, and subjects in the middle are randomized. TBDs are intermediate between regression discontinuity designs (RDDs) and randomized controlled trials (RCTs) by allowing a tradeoff between the resource allocation efficiency of an RDD and the statistical efficiency of an RCT. We study a model where the expected response is one multivariate regression for treated subjects and another for control subjects. For given covariates, we show how to use convex optimization to choose treatment probabilities that optimize a D-optimality criterion. We can incorporate a variety of constraints motivated by economic and ethical considerations. In our model, D-optimality for the treatment effect coincides with D-optimality for the whole regression, and without economic constraints, an RCT is g
    
[^3]: 克服异方差PCA中病态问题的缩减算法

    Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA. (arXiv:2303.06198v1 [math.ST])

    [http://arxiv.org/abs/2303.06198](http://arxiv.org/abs/2303.06198)

    本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。

    This paper proposes a novel algorithm, called Deflated-HeteroPCA, that overcomes the curse of ill-conditioning in heteroskedastic PCA while achieving near-optimal and condition-number-free theoretical guarantees.

    本文关注于从受污染的数据中估计低秩矩阵X*的列子空间。当存在异方差噪声和不平衡的维度（即n2 >> n1）时，如何在容纳最广泛的信噪比范围的同时获得最佳的统计精度变得特别具有挑战性。虽然最先进的算法HeteroPCA成为解决这个问题的强有力的解决方案，但它遭受了“病态问题的诅咒”，即随着X*的条件数增长，其性能会下降。为了克服这个关键问题而不影响允许的信噪比范围，我们提出了一种新的算法，称为缩减异方差PCA，它在$\ell_2$和$\ell_{2,\infty}$统计精度方面实现了近乎最优和无条件数的理论保证。所提出的算法将谱分成两部分

    This paper is concerned with estimating the column subspace of a low-rank matrix $\boldsymbol{X}^\star \in \mathbb{R}^{n_1\times n_2}$ from contaminated data. How to obtain optimal statistical accuracy while accommodating the widest range of signal-to-noise ratios (SNRs) becomes particularly challenging in the presence of heteroskedastic noise and unbalanced dimensionality (i.e., $n_2\gg n_1$). While the state-of-the-art algorithm $\textsf{HeteroPCA}$ emerges as a powerful solution for solving this problem, it suffers from "the curse of ill-conditioning," namely, its performance degrades as the condition number of $\boldsymbol{X}^\star$ grows. In order to overcome this critical issue without compromising the range of allowable SNRs, we propose a novel algorithm, called $\textsf{Deflated-HeteroPCA}$, that achieves near-optimal and condition-number-free theoretical guarantees in terms of both $\ell_2$ and $\ell_{2,\infty}$ statistical accuracy. The proposed algorithm divides the spectrum
    
[^4]: MARS via LASSO.（arXiv:2111.11694v2 [math.ST] 已更新）

    MARS via LASSO. (arXiv:2111.11694v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2111.11694](http://arxiv.org/abs/2111.11694)

    本文提出了一种自然lasso变体的MARS方法，通过减少对维度的依赖来获得收敛率，并与使用平滑性约束的非参数估计技术联系在一起。

    

    多元自适应回归样条（Multivariate Adaptive Regression Splines，MARS）是Friedman在1991年提出的一种非参数回归方法。MARS将简单的非线性和非加性函数拟合到回归数据上。本文提出并研究了MARS方法的一种自然lasso变体。我们的方法是基于最小二乘估计，通过考虑MARS基础函数的无限维线性组合并强加基于变分的复杂度约束条件来获得函数的凸类。虽然我们的估计是定义为无限维优化问题的解，但其可以通过有限维凸优化来计算。在一些标准设计假设下，我们证明了我们的估计器仅在维度上对数收敛，因此在一定程度上避免了通常的维度灾难。我们还表明，我们的方法自然地与基于平滑性约束的非参数估计技术相联系。

    Multivariate adaptive regression splines (MARS) is a popular method for nonparametric regression introduced by Friedman in 1991. MARS fits simple nonlinear and non-additive functions to regression data. We propose and study a natural lasso variant of the MARS method. Our method is based on least squares estimation over a convex class of functions obtained by considering infinite-dimensional linear combinations of functions in the MARS basis and imposing a variation based complexity constraint. Our estimator can be computed via finite-dimensional convex optimization, although it is defined as a solution to an infinite-dimensional optimization problem. Under a few standard design assumptions, we prove that our estimator achieves a rate of convergence that depends only logarithmically on dimension and thus avoids the usual curse of dimensionality to some extent. We also show that our method is naturally connected to nonparametric estimation techniques based on smoothness constraints. We i
    
[^5]: 部分知识下的最优打分规则设计

    Optimal Scoring Rule Design under Partial Knowledge. (arXiv:2107.07420v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2107.07420](http://arxiv.org/abs/2107.07420)

    本文研究了在委托人对代理人的信号分布部分了解的情况下，最优打分规则的设计问题。作者提出了一个最大最小优化的框架，来最大化在代理人信号分布的集合中最坏情况下回报的增加。对于有限集合，提出了高效的算法；对于无限集合，提出了完全多项式时间逼近方案。

    

    本文研究了当委托人对代理人的信号分布部分了解时，最优适当打分规则的设计。最近的工作表明，在委托人完全了解代理人的信号分布的假设下，可以确定增加代理人回报的最大适当打分规则，当代理人选择访问昂贵信号以完善其先验预测的后验信念时。在我们的设置中，委托人只知道代理人的信号分布属于一组分布中的某个。我们将打分规则设计问题制定为最大最小优化问题，最大化分布集合中最坏情况下回报的增加。当分布集合有限时，我们提出了一种高效的算法来计算最优打分规则，并设计了一种完全多项式时间逼近方案，适用于各种无限集合的分布。我们进一步指出，广泛使用的打分规则，如二次方打分规则。

    This paper studies the design of optimal proper scoring rules when the principal has partial knowledge of an agent's signal distribution. Recent work characterizes the proper scoring rules that maximize the increase of an agent's payoff when the agent chooses to access a costly signal to refine a posterior belief from her prior prediction, under the assumption that the agent's signal distribution is fully known to the principal. In our setting, the principal only knows about a set of distributions where the agent's signal distribution belongs. We formulate the scoring rule design problem as a max-min optimization that maximizes the worst-case increase in payoff across the set of distributions.  We propose an efficient algorithm to compute an optimal scoring rule when the set of distributions is finite, and devise a fully polynomial-time approximation scheme that accommodates various infinite sets of distributions. We further remark that widely used scoring rules, such as the quadratic 
    

