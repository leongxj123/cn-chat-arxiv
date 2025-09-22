# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modelling with Discretized Variables](https://arxiv.org/abs/2403.15220) | 该文提出了一种通过离散化方法来实现回归参数点标识的模型，同时保护数据机密性，适用于解决面板数据中多元线性回归参数估计的方法。 |
| [^2] | [Causal inference for the expected number of recurrent events in the presence of a terminal event.](http://arxiv.org/abs/2306.16571) | 在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。 |

# 详细

[^1]: 使用离散变量建模

    Modelling with Discretized Variables

    [https://arxiv.org/abs/2403.15220](https://arxiv.org/abs/2403.15220)

    该文提出了一种通过离散化方法来实现回归参数点标识的模型，同时保护数据机密性，适用于解决面板数据中多元线性回归参数估计的方法。

    

    本文讨论了在计量经济模型中，因变量、一些解释变量或两者均以被截断的区间数据的形式观测到的情况。这种离散化通常是由于敏感变量如收入的保密性造成的。使用这些变量的模型无法对回归参数进行点识别，因为条件矩未知，这导致文献使用区间估计。在这里，我们提出了一种离散化方法，通过该方法可以点识别回归参数，同时保护数据的机密性。我们展示了OLS估计器在面板数据的多元线性回归中的渐近性质。理论发现得到蒙特卡洛实验的支持，并通过应用到澳大利亚性别工资差距上进行了说明。

    arXiv:2403.15220v1 Announce Type: new  Abstract: This paper deals with econometric models in which the dependent variable, some explanatory variables, or both are observed as censored interval data. This discretization often happens due to confidentiality of sensitive variables like income. Models using these variables cannot point identify regression parameters as the conditional moments are unknown, which led the literature to use interval estimates. Here, we propose a discretization method through which the regression parameters can be point identified while preserving data confidentiality. We demonstrate the asymptotic properties of the OLS estimator for the parameters in multivariate linear regressions for cross-sectional data. The theoretical findings are supported by Monte Carlo experiments and illustrated with an application to the Australian gender wage gap.
    
[^2]: 在存在终结事件的情况下，关于经常性事件的因果推断

    Causal inference for the expected number of recurrent events in the presence of a terminal event. (arXiv:2306.16571v1 [stat.ME])

    [http://arxiv.org/abs/2306.16571](http://arxiv.org/abs/2306.16571)

    在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。

    

    我们研究了在存在终结事件的情况下，关于经常性事件的因果推断和高效估计。我们将估计目标定义为包括经常性事件的预期数量以及在一系列里程碑时间点处评估的失败生存函数的向量。我们在右截尾和因果选择的情况下确定了估计目标，作为观察数据的功能性，推导了非参数效率界限，并提出了一种多重鲁棒估计器，该估计器达到了界限，并允许非参数估计辅助参数。在整个过程中，我们对失败、截尾或观察数据的概率分布没有做绝对连续性的假设。此外，当分割分布已知时，我们导出了影响函数的类别，并回顾了已发表估计器如何属于该类别。在此过程中，我们强调了因果生命周期中一些有趣的不一致性。

    We study causal inference and efficient estimation for the expected number of recurrent events in the presence of a terminal event. We define our estimand as the vector comprising both the expected number of recurrent events and the failure survival function evaluated along a sequence of landmark times. We identify the estimand in the presence of right-censoring and causal selection as an observed data functional under coarsening at random, derive the nonparametric efficiency bound, and propose a multiply-robust estimator that achieves the bound and permits nonparametric estimation of nuisance parameters. Throughout, no absolute continuity assumption is made on the underlying probability distributions of failure, censoring, or the observed data. Additionally, we derive the class of influence functions when the coarsening distribution is known and review how published estimators may belong to the class. Along the way, we highlight some interesting inconsistencies in the causal lifetime 
    

