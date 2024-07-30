# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Protected Spatial Autoregressive Model](https://arxiv.org/abs/2403.16773) | 提出了一种隐私保护的空间自回归模型，引入了噪声响应和协变量以满足隐私保护要求，并开发了纠正由噪声引入的偏差的技术。 |
| [^2] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |
| [^3] | [Nonparametric Causal Decomposition of Group Disparities.](http://arxiv.org/abs/2306.16591) | 本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。 |

# 详细

[^1]: 隐私保护的空间自回归模型

    Privacy-Protected Spatial Autoregressive Model

    [https://arxiv.org/abs/2403.16773](https://arxiv.org/abs/2403.16773)

    提出了一种隐私保护的空间自回归模型，引入了噪声响应和协变量以满足隐私保护要求，并开发了纠正由噪声引入的偏差的技术。

    

    空间自回归（SAR）模型是研究网络效应的重要工具。然而，随着对数据隐私的重视增加，数据提供者经常实施隐私保护措施，使传统的SAR模型变得不适用。在本研究中，我们介绍了一种带有添加噪声响应和协变量的隐私保护的SAR模型，以满足隐私保护要求。然而，在这种情况下，由于无法建立似然函数，传统的拟最大似然估计变得不可行。为了解决这个问题，我们首先考虑了只有噪声添加响应的似然函数的显式表达。然而，由于协变量中的噪声，导数是有偏的。因此，我们开发了可以纠正噪声引入的偏差的技术。相应地，提出了一种类似牛顿-拉弗森的算法来获得估计量，从而导致一个修正的似然估计量。

    arXiv:2403.16773v1 Announce Type: cross  Abstract: Spatial autoregressive (SAR) models are important tools for studying network effects. However, with an increasing emphasis on data privacy, data providers often implement privacy protection measures that make classical SAR models inapplicable. In this study, we introduce a privacy-protected SAR model with noise-added response and covariates to meet privacy-protection requirements. However, in this scenario, the traditional quasi-maximum likelihood estimator becomes infeasible because the likelihood function cannot be formulated. To address this issue, we first consider an explicit expression for the likelihood function with only noise-added responses. However, the derivatives are biased owing to the noise in the covariates. Therefore, we develop techniques that can correct the biases introduced by noise. Correspondingly, a Newton-Raphson-type algorithm is proposed to obtain the estimator, leading to a corrected likelihood estimator. To
    
[^2]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    
[^3]: 非参数因果分解组差异

    Nonparametric Causal Decomposition of Group Disparities. (arXiv:2306.16591v1 [stat.ME])

    [http://arxiv.org/abs/2306.16591](http://arxiv.org/abs/2306.16591)

    本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。

    

    我们提出了一个因果框架来将结果中的组差异分解为中间处理变量。我们的框架捕捉了基线潜在结果、处理前沿、平均处理效应和处理选择的组差异的贡献。这个框架以反事实的方式进行了数学表达，并且能够方便地指导政策干预。特别是，针对不同的处理选择进行的分解部分是特别新颖的，揭示了一种解释和改善差异的新机制。这个框架以因果术语重新定义了经典的Kitagawa-Blinder-Oaxaca分解，通过解释组差异而不是组效应来补充了因果中介分析，并解决了近期随机等化分解的概念困难。我们还提供了一个条件分解，允许研究人员在定义评估和相应的干预措施时纳入协变量。

    We propose a causal framework for decomposing a group disparity in an outcome in terms of an intermediate treatment variable. Our framework captures the contributions of group differences in baseline potential outcome, treatment prevalence, average treatment effect, and selection into treatment. This framework is counterfactually formulated and readily informs policy interventions. The decomposition component for differential selection into treatment is particularly novel, revealing a new mechanism for explaining and ameliorating disparities. This framework reformulates the classic Kitagawa-Blinder-Oaxaca decomposition in causal terms, supplements causal mediation analysis by explaining group disparities instead of group effects, and resolves conceptual difficulties of recent random equalization decompositions. We also provide a conditional decomposition that allows researchers to incorporate covariates in defining the estimands and corresponding interventions. We develop nonparametric
    

