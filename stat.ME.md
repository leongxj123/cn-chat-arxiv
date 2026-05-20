# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile](https://arxiv.org/abs/2403.20200) | 研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。 |
| [^2] | [Testing the identification of causal effects in observational data.](http://arxiv.org/abs/2203.15890) | 本研究提出一种机器学习方法用于检测观测数据中因果效应的识别，并且连带着提出一种工具变量和协变量的可测试条件，这为治疗效果的评估提供了途径。 |

# 详细

[^1]: 对具有方差轮廓的非独立同分布数据的岭回归进行高维分析

    High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile

    [https://arxiv.org/abs/2403.20200](https://arxiv.org/abs/2403.20200)

    研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。

    

    针对独立但非独立同分布数据，我们提出研究高维回归模型。假设观测到的预测变量集合是带有方差轮廓的随机矩阵，并且其维度以相应速率增长。在假设随机效应模型的情况下，我们研究了具有这种方差轮廓的岭估计器的线性回归的预测风险。在这种设置下，我们提供了该风险的确定性等价物以及岭估计器的自由度。对于某些方差轮廓类别，我们的工作突出了在岭正则化参数趋于零时，高维回归中的最小模最小二乘估计器出现双谷现象。我们还展示了一些方差轮廓f...

    arXiv:2403.20200v1 Announce Type: cross  Abstract: High-dimensional linear regression has been thoroughly studied in the context of independent and identically distributed data. We propose to investigate high-dimensional regression models for independent but non-identically distributed data. To this end, we suppose that the set of observed predictors (or features) is a random matrix with a variance profile and with dimensions growing at a proportional rate. Assuming a random effect model, we study the predictive risk of the ridge estimator for linear regression with such a variance profile. In this setting, we provide deterministic equivalents of this risk and of the degree of freedom of the ridge estimator. For certain class of variance profile, our work highlights the emergence of the well-known double descent phenomenon in high-dimensional regression for the minimum norm least-squares estimator when the ridge regularization parameter goes to zero. We also exhibit variance profiles f
    
[^2]: 在观测数据中检验因果效应的识别

    Testing the identification of causal effects in observational data. (arXiv:2203.15890v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.15890](http://arxiv.org/abs/2203.15890)

    本研究提出一种机器学习方法用于检测观测数据中因果效应的识别，并且连带着提出一种工具变量和协变量的可测试条件，这为治疗效果的评估提供了途径。

    

    本研究展示了一个可测试的条件，用于在观测数据中识别治疗对结果的因果效应，该条件基于两组变量：需要控制的观测协变量和被怀疑的工具变量。在实证应用中常见的因果结构下，被怀疑的工具变量与结果（在给定治疗和协变量的条件下）的可测试条件独立性有两个含义。第一，工具变量是有效的，即它不直接影响结果（除了通过治疗）并在协变量条件下未受到混淆。第二，治疗在协变量条件下不受混淆，因此治疗效应得以确定。我们建议使用基于机器学习方法的条件独立性测试，以数据驱动的方式考虑协变量，并在模拟研究中研究其渐近行为和有限样本性能。我们还将我们的方法应用于真实数据，以说明其适用性。

    This study demonstrates the existence of a testable condition for the identification of the causal effect of a treatment on an outcome in observational data, which relies on two sets of variables: observed covariates to be controlled for and a suspected instrument. Under a causal structure commonly found in empirical applications, the testable conditional independence of the suspected instrument and the outcome given the treatment and the covariates has two implications. First, the instrument is valid, i.e. it does not directly affect the outcome (other than through the treatment) and is unconfounded conditional on the covariates. Second, the treatment is unconfounded conditional on the covariates such that the treatment effect is identified. We suggest tests of this conditional independence based on machine learning methods that account for covariates in a data-driven way and investigate their asymptotic behavior and finite sample performance in a simulation study. We also apply our t
    

