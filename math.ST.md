# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Yurinskii's Coupling for Martingales](https://arxiv.org/abs/2210.00362) | Yurinskii的耦合方法在$\ell^p$-范数下提供了更弱条件下的逼近马丁格尔，同时引入了更一般的高斯混合分布，并提供了第三阶耦合方法以在某些情况下获得更紧密的逼近。 |
| [^2] | [Policy Learning with Distributional Welfare.](http://arxiv.org/abs/2311.15878) | 本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。 |
| [^3] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |
| [^4] | [Batches Stabilize the Minimum Norm Risk in High Dimensional Overparameterized Linear Regression.](http://arxiv.org/abs/2306.08432) | 本文研究了将数据分成批次的学习算法，在高维超参数线性回归模型中提供了隐式正则化，通过适当的批量大小选择，稳定了风险行为，消除了插值点处的膨胀和双峰现象 |

# 详细

[^1]: Yurinskii的马丁格尔耦合

    Yurinskii's Coupling for Martingales

    [https://arxiv.org/abs/2210.00362](https://arxiv.org/abs/2210.00362)

    Yurinskii的耦合方法在$\ell^p$-范数下提供了更弱条件下的逼近马丁格尔，同时引入了更一般的高斯混合分布，并提供了第三阶耦合方法以在某些情况下获得更紧密的逼近。

    

    Yurinskii的耦合是数学统计和应用概率中一种常用的非渐近分布分析理论工具，提供了在易于验证条件下具有显式误差界限的高斯强逼近。最初在独立随机向量和为的$\ell^2$-范数中陈述，最近已将其扩展到$1 \leq p \leq \infty$时的$\ell^p$-范数，以及在某些强条件下的$\ell^2$-范数的向量值鞅。我们的主要结果是在远比之前施加的条件更弱的情况下，在$\ell^p$-范数下提供了逼近马丁格尔的Yurinskii耦合。我们的公式进一步允许耦合变量遵循更一般的高斯混合分布，并且我们提供了一种新颖的第三阶耦合方法，在某些情况下提供更紧密的逼近。我们将我们的主要结果专门应用于混合马丁格尔，马丁格尔和其他情况。

    arXiv:2210.00362v2 Announce Type: replace-cross  Abstract: Yurinskii's coupling is a popular theoretical tool for non-asymptotic distributional analysis in mathematical statistics and applied probability, offering a Gaussian strong approximation with an explicit error bound under easily verified conditions. Originally stated in $\ell^2$-norm for sums of independent random vectors, it has recently been extended both to the $\ell^p$-norm, for $1 \leq p \leq \infty$, and to vector-valued martingales in $\ell^2$-norm, under some strong conditions. We present as our main result a Yurinskii coupling for approximate martingales in $\ell^p$-norm, under substantially weaker conditions than those previously imposed. Our formulation further allows for the coupling variable to follow a more general Gaussian mixture distribution, and we provide a novel third-order coupling method which gives tighter approximations in certain settings. We specialize our main result to mixingales, martingales, and in
    
[^2]: 分配福利的政策学习

    Policy Learning with Distributional Welfare. (arXiv:2311.15878v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.15878](http://arxiv.org/abs/2311.15878)

    本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。

    

    本文探讨了针对分配福利的最优治疗分配策略。大部分关于治疗选择的文献都考虑了基于条件平均治疗效应（ATE）的功利福利。虽然平均福利是直观的，但在个体异质化（例如，存在离群值）情况下可能会产生不理想的分配 - 这正是个性化治疗引入的原因之一。这个观察让我们提出了一种根据个体治疗效应的条件分位数（QoTE）来分配治疗的最优策略。根据分位数概率的选择，这个准则可以适应谨慎或粗心的决策者。确定QoTE的挑战在于其需要对反事实结果的联合分布有所了解，但即使使用实验数据，通常也很难恢复出来。因此，我们介绍了鲁棒的最小最大化策略

    In this paper, we explore optimal treatment allocation policies that target distributional welfare. Most literature on treatment choice has considered utilitarian welfare based on the conditional average treatment effect (ATE). While average welfare is intuitive, it may yield undesirable allocations especially when individuals are heterogeneous (e.g., with outliers) - the very reason individualized treatments were introduced in the first place. This observation motivates us to propose an optimal policy that allocates the treatment based on the conditional quantile of individual treatment effects (QoTE). Depending on the choice of the quantile probability, this criterion can accommodate a policymaker who is either prudent or negligent. The challenge of identifying the QoTE lies in its requirement for knowledge of the joint distribution of the counterfactual outcomes, which is generally hard to recover even with experimental data. Therefore, we introduce minimax policies that are robust 
    
[^3]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    
[^4]: 批次使高维超参数线性回归的最小规范风险稳定

    Batches Stabilize the Minimum Norm Risk in High Dimensional Overparameterized Linear Regression. (arXiv:2306.08432v1 [cs.LG])

    [http://arxiv.org/abs/2306.08432](http://arxiv.org/abs/2306.08432)

    本文研究了将数据分成批次的学习算法，在高维超参数线性回归模型中提供了隐式正则化，通过适当的批量大小选择，稳定了风险行为，消除了插值点处的膨胀和双峰现象

    

    将数据分成批次的学习算法在许多机器学习应用中很常见，通常在计算效率和性能之间提供有用的权衡。本文通过具有各向同性高斯特征的最小规范超参数线性回归模型的视角来研究批量分区的好处。我们建议最小规范估计量的自然小批量版本，并推导出其二次风险的上界，表明其与噪声水平以及过度参数化比例成反比，对于最佳批量大小的选择。与最小规范相比，我们的估计器具有稳定的风险行为，其在过度参数化比例上单调递增，消除了插值点处的膨胀和双峰现象。有趣的是，我们观察到批处理所提供的隐式正则化在一定程度上可以通过特征重叠来解释。

    Learning algorithms that divide the data into batches are prevalent in many machine-learning applications, typically offering useful trade-offs between computational efficiency and performance. In this paper, we examine the benefits of batch-partitioning through the lens of a minimum-norm overparameterized linear regression model with isotropic Gaussian features. We suggest a natural small-batch version of the minimum-norm estimator, and derive an upper bound on its quadratic risk, showing it is inversely proportional to the noise level as well as to the overparameterization ratio, for the optimal choice of batch size. In contrast to minimum-norm, our estimator admits a stable risk behavior that is monotonically increasing in the overparameterization ratio, eliminating both the blowup at the interpolation point and the double-descent phenomenon. Interestingly, we observe that this implicit regularization offered by the batch partition is partially explained by feature overlap between t
    

