# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Two-Stage Nuisance Function Estimation for Causal Mediation Analysis](https://arxiv.org/abs/2404.00735) | 通过两阶段估计策略，该研究提出了一种针对因果中介分析中干扰函数的方法，旨在根据其在偏差结构中的作用来估计干扰函数。 |
| [^2] | [A Double Machine Learning Approach to Combining Experimental and Observational Data.](http://arxiv.org/abs/2307.01449) | 这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。 |
| [^3] | [Causal inference for the expected number of recurrent events in the presence of a terminal event.](http://arxiv.org/abs/2306.16571) | 在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。 |

# 详细

[^1]: 因果中介分析的两阶段干扰函数估计

    Two-Stage Nuisance Function Estimation for Causal Mediation Analysis

    [https://arxiv.org/abs/2404.00735](https://arxiv.org/abs/2404.00735)

    通过两阶段估计策略，该研究提出了一种针对因果中介分析中干扰函数的方法，旨在根据其在偏差结构中的作用来估计干扰函数。

    

    在使用基于影响函数的中介功能估计器估计直接和间接因果效应时，了解应该关注治疗、中介和结果的哪些方面是至关重要的。具体而言，将它们视为干扰函数，并试图尽可能准确地拟合这些干扰函数并不一定是最好的方法。在这项工作中，我们提出了一种针对干扰函数的两阶段估计策略，该策略根据干扰函数在影响函数的中介功能估计器的偏差结构中发挥的作用来估计干扰函数。我们对所提出方法进行了稳健性分析，以及参数估计器的一致性和渐近正态性的充分条件。

    arXiv:2404.00735v1 Announce Type: cross  Abstract: When estimating the direct and indirect causal effects using the influence function-based estimator of the mediation functional, it is crucial to understand what aspects of the treatment, the mediator, and the outcome mean mechanisms should be focused on. Specifically, considering them as nuisance functions and attempting to fit these nuisance functions as accurate as possible is not necessarily the best approach to take. In this work, we propose a two-stage estimation strategy for the nuisance functions that estimates the nuisance functions based on the role they play in the structure of the bias of the influence function-based estimator of the mediation functional. We provide robustness analysis of the proposed method, as well as sufficient conditions for consistency and asymptotic normality of the estimator of the parameter of interest.
    
[^2]: 将实验数据与观测数据结合的双机器学习方法

    A Double Machine Learning Approach to Combining Experimental and Observational Data. (arXiv:2307.01449v1 [stat.ME])

    [http://arxiv.org/abs/2307.01449](http://arxiv.org/abs/2307.01449)

    这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。

    

    实验和观测研究通常由于无法测试的假设而缺乏有效性。我们提出了一种双机器学习方法，将实验和观测研究结合起来，使从业人员能够测试假设违反情况并一致估计处理效应。我们的框架在较轻的假设下测试外部效度和可忽视性的违反情况。当只有一个假设被违反时，我们提供半参数高效的处理效应估计器。然而，我们的无免费午餐定理强调了准确识别违反的假设对一致的处理效应估计的必要性。我们通过三个实际案例研究展示了我们方法的适用性，并突出了其在实际环境中的相关性。

    Experimental and observational studies often lack validity due to untestable assumptions. We propose a double machine learning approach to combine experimental and observational studies, allowing practitioners to test for assumption violations and estimate treatment effects consistently. Our framework tests for violations of external validity and ignorability under milder assumptions. When only one assumption is violated, we provide semi-parametrically efficient treatment effect estimators. However, our no-free-lunch theorem highlights the necessity of accurately identifying the violated assumption for consistent treatment effect estimation. We demonstrate the applicability of our approach in three real-world case studies, highlighting its relevance for practical settings.
    
[^3]: 在存在终结事件的情况下，关于经常性事件的因果推断

    Causal inference for the expected number of recurrent events in the presence of a terminal event. (arXiv:2306.16571v1 [stat.ME])

    [http://arxiv.org/abs/2306.16571](http://arxiv.org/abs/2306.16571)

    在存在终结事件的情况下，研究经常性事件的因果推断和高效估计，提出了一种基于乘法鲁棒估计的方法，不依赖于分布假设，并指出了一些有趣的因果生命周期中的不一致性。

    

    我们研究了在存在终结事件的情况下，关于经常性事件的因果推断和高效估计。我们将估计目标定义为包括经常性事件的预期数量以及在一系列里程碑时间点处评估的失败生存函数的向量。我们在右截尾和因果选择的情况下确定了估计目标，作为观察数据的功能性，推导了非参数效率界限，并提出了一种多重鲁棒估计器，该估计器达到了界限，并允许非参数估计辅助参数。在整个过程中，我们对失败、截尾或观察数据的概率分布没有做绝对连续性的假设。此外，当分割分布已知时，我们导出了影响函数的类别，并回顾了已发表估计器如何属于该类别。在此过程中，我们强调了因果生命周期中一些有趣的不一致性。

    We study causal inference and efficient estimation for the expected number of recurrent events in the presence of a terminal event. We define our estimand as the vector comprising both the expected number of recurrent events and the failure survival function evaluated along a sequence of landmark times. We identify the estimand in the presence of right-censoring and causal selection as an observed data functional under coarsening at random, derive the nonparametric efficiency bound, and propose a multiply-robust estimator that achieves the bound and permits nonparametric estimation of nuisance parameters. Throughout, no absolute continuity assumption is made on the underlying probability distributions of failure, censoring, or the observed data. Additionally, we derive the class of influence functions when the coarsening distribution is known and review how published estimators may belong to the class. Along the way, we highlight some interesting inconsistencies in the causal lifetime 
    

