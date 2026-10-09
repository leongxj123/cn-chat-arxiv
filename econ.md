# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Debiased Machine Learning when Nuisance Parameters Appear in Indicator Functions](https://arxiv.org/abs/2403.15934) | 本文提出了平滑指示函数的方法，并为这类模型开发了渐近分布理论，展现了偏差和方差之间的折衷关系，并研究了如何选择最优的平滑程度参数。 |
| [^2] | [Trimmed Mean Group Estimation of Average Treatment Effects in Ultra Short T Panels under Correlated Heterogeneity.](http://arxiv.org/abs/2310.11680) | 本文在相关异质性下，提出了一种修剪均值组（TMG）估计器，可以在面板数据时间维度很小的情况下以不规则的速度保持一致性。该方法具有较好的性质和性能，并提供了相关异质性的检验方法。通过实证应用展示了该方法的实用性。 |

# 详细

[^1]: 当指示函数中出现干扰参数时的去偏机器学习

    Debiased Machine Learning when Nuisance Parameters Appear in Indicator Functions

    [https://arxiv.org/abs/2403.15934](https://arxiv.org/abs/2403.15934)

    本文提出了平滑指示函数的方法，并为这类模型开发了渐近分布理论，展现了偏差和方差之间的折衷关系，并研究了如何选择最优的平滑程度参数。

    

    本文研究了当指示函数中出现干扰参数时的去偏机器学习。一个重要的例子是在最优治疗分配规则下最大化平均福利。为了对感兴趣的参数进行渐近有效推断，当前有关去偏机器学习的文献依赖于矩条件内部函数的Gateaux可微性，当指示函数中出现干扰参数时，这种可微性不成立。本文提出了平滑指示函数的方法，并为这类模型开发了渐近分布理论。所提估计量的渐近行为表现出由于平滑而产生的偏差和方差之间的折衷。我们研究了如何选择控制平滑程度的参数以最小化渐近均方误差的上限。蒙特卡洛模拟支持了渐近分布理论，并且实证结果

    arXiv:2403.15934v1 Announce Type: new  Abstract: This paper studies debiased machine learning when nuisance parameters appear in indicator functions. An important example is maximized average welfare under optimal treatment assignment rules. For asymptotically valid inference for a parameter of interest, the current literature on debiased machine learning relies on Gateaux differentiability of the functions inside moment conditions, which does not hold when nuisance parameters appear in indicator functions. In this paper, we propose smoothing the indicator functions, and develop an asymptotic distribution theory for this class of models. The asymptotic behavior of the proposed estimator exhibits a trade-off between bias and variance due to smoothing. We study how a parameter which controls the degree of smoothing can be chosen optimally to minimize an upper bound of the asymptotic mean squared error. A Monte Carlo simulation supports the asymptotic distribution theory, and an empirical
    
[^2]: 在相关异质性下，短期面板中关于平均处理效应的修剪均值组估计方法

    Trimmed Mean Group Estimation of Average Treatment Effects in Ultra Short T Panels under Correlated Heterogeneity. (arXiv:2310.11680v1 [econ.EM])

    [http://arxiv.org/abs/2310.11680](http://arxiv.org/abs/2310.11680)

    本文在相关异质性下，提出了一种修剪均值组（TMG）估计器，可以在面板数据时间维度很小的情况下以不规则的速度保持一致性。该方法具有较好的性质和性能，并提供了相关异质性的检验方法。通过实证应用展示了该方法的实用性。

    

    在相关异质性下，常用的两路固定效应估计方法存在偏差并可能导致误导性推断。本文提出了一种新的修剪均值组估计器（TMG estimator），即使面板的时间维度与回归变量数目一样小，也能以不规则的n^{1/3}速度保持一致性。本文还提供了适用于具有时间效应的面板的扩展方法，并提出了一种相关异质性的豪斯曼式检验。通过蒙特卡洛实验，研究了TMG估计器（带有和不带有时间效应）在小样本情况下的性质，结果表明其性能令人满意，优于文献中提出的其他修剪估计器。同时，所提出的相关异质性检验显示出正确的大小和令人满意的功效。通过实证应用，展示了TMG方法的实用性。

    Under correlated heterogeneity, the commonly used two-way fixed effects estimator is biased and can lead to misleading inference. This paper proposes a new trimmed mean group (TMG) estimator which is consistent at the irregular rate of n^{1/3} even if the time dimension of the panel is as small as the number of its regressors. Extensions to panels with time effects are provided, and a Hausman-type test of correlated heterogeneity is proposed. Small sample properties of the TMG estimator (with and without time effects) are investigated by Monte Carlo experiments and shown to be satisfactory and perform better than other trimmed estimators proposed in the literature. The proposed test of correlated heterogeneity is also shown to have the correct size and satisfactory power. The utility of the TMG approach is illustrated with an empirical application.
    

