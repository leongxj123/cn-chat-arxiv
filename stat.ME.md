# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding and avoiding the "weights of regression": Heterogeneous effects, misspecification, and longstanding solutions](https://arxiv.org/abs/2403.03299) | 处理的回归系数通常不等于平均处理效应（ATE），且可能不是直接科学或政策感兴趣的数量。研究人员提出各种解释、边界和诊断辅助工具来解释这种差异。 |
| [^2] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |

# 详细

[^1]: 理解和避免“回归权重”：异质效应、误设和长久解决方案

    Understanding and avoiding the "weights of regression": Heterogeneous effects, misspecification, and longstanding solutions

    [https://arxiv.org/abs/2403.03299](https://arxiv.org/abs/2403.03299)

    处理的回归系数通常不等于平均处理效应（ATE），且可能不是直接科学或政策感兴趣的数量。研究人员提出各种解释、边界和诊断辅助工具来解释这种差异。

    

    许多领域的研究人员努力通过在处理（D）和观察到的混杂因素（X）上对结果数据（Y）进行回归来估计治疗效应。即使不存在未观察到的混杂因素，处理的回归系数也会报告分层特定处理效应的加权平均值。当无法排除异质处理效应时，得到的系数通常不等于平均处理效应（ATE），也不太可能是直接科学或政策感兴趣的数量。系数与ATE之间的差异导致研究人员提出各种解释、边界和诊断辅助工具。我们注意到，在处理效应在X中是异质的时，对Y关于D和X的线性回归可能存在误设。回归的“权重”，对此我们提供了一种新的...

    arXiv:2403.03299v1 Announce Type: cross  Abstract: Researchers in many fields endeavor to estimate treatment effects by regressing outcome data (Y) on a treatment (D) and observed confounders (X). Even absent unobserved confounding, the regression coefficient on the treatment reports a weighted average of strata-specific treatment effects (Angrist, 1998). Where heterogeneous treatment effects cannot be ruled out, the resulting coefficient is thus not generally equal to the average treatment effect (ATE), and is unlikely to be the quantity of direct scientific or policy interest. The difference between the coefficient and the ATE has led researchers to propose various interpretational, bounding, and diagnostic aids (Humphreys, 2009; Aronow and Samii, 2016; Sloczynski, 2022; Chattopadhyay and Zubizarreta, 2023). We note that the linear regression of Y on D and X can be misspecified when the treatment effect is heterogeneous in X. The "weights of regression", for which we provide a new (m
    
[^2]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    

