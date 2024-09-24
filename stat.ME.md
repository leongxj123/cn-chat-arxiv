# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rapid Bayesian identification of sparse nonlinear dynamics from scarce and noisy data](https://arxiv.org/abs/2402.15357) | 提出了一种快速的概率框架，称为贝叶斯-SINDy，用于从有限且嘈杂数据中学习正确的模型方程，并且对参数估计中的不确定性进行量化，特别适用于生物数据和实时系统识别。 |
| [^2] | [Yurinskii's Coupling for Martingales](https://arxiv.org/abs/2210.00362) | Yurinskii的耦合方法在$\ell^p$-范数下提供了更弱条件下的逼近马丁格尔，同时引入了更一般的高斯混合分布，并提供了第三阶耦合方法以在某些情况下获得更紧密的逼近。 |
| [^3] | [Policy Learning with Distributional Welfare.](http://arxiv.org/abs/2311.15878) | 本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。 |
| [^4] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |

# 详细

[^1]: 从稀疏且嘈杂数据中快速识别稀疏非线性动力学的贝叶斯方法

    Rapid Bayesian identification of sparse nonlinear dynamics from scarce and noisy data

    [https://arxiv.org/abs/2402.15357](https://arxiv.org/abs/2402.15357)

    提出了一种快速的概率框架，称为贝叶斯-SINDy，用于从有限且嘈杂数据中学习正确的模型方程，并且对参数估计中的不确定性进行量化，特别适用于生物数据和实时系统识别。

    

    我们提出了一个快速的概率框架，用于识别控制观测数据动态的微分方程。我们将SINDy方法重新构建到贝叶斯框架中，并使用高斯逼近来加速计算。由此产生的方法，贝叶斯-SINDy，不仅量化了参数估计中的不确定性，而且在从有限且嘈杂数据中学习正确模型时更加稳健。通过使用合成和真实例子，如猞猁-野兔种群动态，我们展示了新框架在学习正确模型方程中的有效性，并比较了其与现有方法的计算和数据效率。由于贝叶斯-SINDy可以快速吸收数据并对噪声具有稳健性，因此特别适用于生物数据和控制中的实时系统识别。其概率框架还使得可以计算信息熵。

    arXiv:2402.15357v1 Announce Type: cross  Abstract: We propose a fast probabilistic framework for identifying differential equations governing the dynamics of observed data. We recast the SINDy method within a Bayesian framework and use Gaussian approximations for the prior and likelihood to speed up computation. The resulting method, Bayesian-SINDy, not only quantifies uncertainty in the parameters estimated but also is more robust when learning the correct model from limited and noisy data. Using both synthetic and real-life examples such as Lynx-Hare population dynamics, we demonstrate the effectiveness of the new framework in learning correct model equations and compare its computational and data efficiency with existing methods. Because Bayesian-SINDy can quickly assimilate data and is robust against noise, it is particularly suitable for biological data and real-time system identification in control. Its probabilistic framework also enables the calculation of information entropy, 
    
[^2]: Yurinskii的马丁格尔耦合

    Yurinskii's Coupling for Martingales

    [https://arxiv.org/abs/2210.00362](https://arxiv.org/abs/2210.00362)

    Yurinskii的耦合方法在$\ell^p$-范数下提供了更弱条件下的逼近马丁格尔，同时引入了更一般的高斯混合分布，并提供了第三阶耦合方法以在某些情况下获得更紧密的逼近。

    

    Yurinskii的耦合是数学统计和应用概率中一种常用的非渐近分布分析理论工具，提供了在易于验证条件下具有显式误差界限的高斯强逼近。最初在独立随机向量和为的$\ell^2$-范数中陈述，最近已将其扩展到$1 \leq p \leq \infty$时的$\ell^p$-范数，以及在某些强条件下的$\ell^2$-范数的向量值鞅。我们的主要结果是在远比之前施加的条件更弱的情况下，在$\ell^p$-范数下提供了逼近马丁格尔的Yurinskii耦合。我们的公式进一步允许耦合变量遵循更一般的高斯混合分布，并且我们提供了一种新颖的第三阶耦合方法，在某些情况下提供更紧密的逼近。我们将我们的主要结果专门应用于混合马丁格尔，马丁格尔和其他情况。

    arXiv:2210.00362v2 Announce Type: replace-cross  Abstract: Yurinskii's coupling is a popular theoretical tool for non-asymptotic distributional analysis in mathematical statistics and applied probability, offering a Gaussian strong approximation with an explicit error bound under easily verified conditions. Originally stated in $\ell^2$-norm for sums of independent random vectors, it has recently been extended both to the $\ell^p$-norm, for $1 \leq p \leq \infty$, and to vector-valued martingales in $\ell^2$-norm, under some strong conditions. We present as our main result a Yurinskii coupling for approximate martingales in $\ell^p$-norm, under substantially weaker conditions than those previously imposed. Our formulation further allows for the coupling variable to follow a more general Gaussian mixture distribution, and we provide a novel third-order coupling method which gives tighter approximations in certain settings. We specialize our main result to mixingales, martingales, and in
    
[^3]: 分配福利的政策学习

    Policy Learning with Distributional Welfare. (arXiv:2311.15878v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.15878](http://arxiv.org/abs/2311.15878)

    本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。

    

    本文探讨了针对分配福利的最优治疗分配策略。大部分关于治疗选择的文献都考虑了基于条件平均治疗效应（ATE）的功利福利。虽然平均福利是直观的，但在个体异质化（例如，存在离群值）情况下可能会产生不理想的分配 - 这正是个性化治疗引入的原因之一。这个观察让我们提出了一种根据个体治疗效应的条件分位数（QoTE）来分配治疗的最优策略。根据分位数概率的选择，这个准则可以适应谨慎或粗心的决策者。确定QoTE的挑战在于其需要对反事实结果的联合分布有所了解，但即使使用实验数据，通常也很难恢复出来。因此，我们介绍了鲁棒的最小最大化策略

    In this paper, we explore optimal treatment allocation policies that target distributional welfare. Most literature on treatment choice has considered utilitarian welfare based on the conditional average treatment effect (ATE). While average welfare is intuitive, it may yield undesirable allocations especially when individuals are heterogeneous (e.g., with outliers) - the very reason individualized treatments were introduced in the first place. This observation motivates us to propose an optimal policy that allocates the treatment based on the conditional quantile of individual treatment effects (QoTE). Depending on the choice of the quantile probability, this criterion can accommodate a policymaker who is either prudent or negligent. The challenge of identifying the QoTE lies in its requirement for knowledge of the joint distribution of the counterfactual outcomes, which is generally hard to recover even with experimental data. Therefore, we introduce minimax policies that are robust 
    
[^4]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    

