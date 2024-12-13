# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Estimation and Inference in Categorical Data](https://arxiv.org/abs/2403.11954) | 提出了一种通用估计器，能够鲁棒地处理分类数据模型的误设，不做任何假设，并且可以应用于任何分类响应模型。 |
| [^2] | [TNDDR: Efficient and doubly robust estimation of COVID-19 vaccine effectiveness under the test-negative design.](http://arxiv.org/abs/2310.04578) | 我们提出了一种高效且双重鲁棒的估计器TNDDR，用于在阴性测试设计下估计COVID-19疫苗的有效性，可有效解决选择偏差问题，并结合机器学习技术进行辅助函数估计。 |

# 详细

[^1]: 在分类数据中的鲁棒估计和推断

    Robust Estimation and Inference in Categorical Data

    [https://arxiv.org/abs/2403.11954](https://arxiv.org/abs/2403.11954)

    提出了一种通用估计器，能够鲁棒地处理分类数据模型的误设，不做任何假设，并且可以应用于任何分类响应模型。

    

    在实证科学中，许多感兴趣的变量是分类的。与任何模型一样，对于分类响应的模型可以被误设，导致估计可能存在较大偏差。一个特别麻烦的误设来源是在问卷调查中的疏忽响应，众所周知这会危及结构方程模型（SEM）和其他基于调查的分析的有效性。我提出了一个旨在对分类响应模型的误设鲁棒的通用估计器。与迄今为止的方法不同，该估计器对分类响应模型的误设程度、大小或类型不做任何假设。所提出的估计器推广了极大似然估计，是强一致的，渐近高斯的，具有与极大似然相同的时间复杂度，并且可以应用于任何分类响应模型。此外，我开发了一个新颖的检验，用于测试一个给定响应是否 ...

    arXiv:2403.11954v1 Announce Type: cross  Abstract: In empirical science, many variables of interest are categorical. Like any model, models for categorical responses can be misspecified, leading to possibly large biases in estimation. One particularly troublesome source of misspecification is inattentive responding in questionnaires, which is well-known to jeopardize the validity of structural equation models (SEMs) and other survey-based analyses. I propose a general estimator that is designed to be robust to misspecification of models for categorical responses. Unlike hitherto approaches, the estimator makes no assumption whatsoever on the degree, magnitude, or type of misspecification. The proposed estimator generalizes maximum likelihood estimation, is strongly consistent, asymptotically Gaussian, has the same time complexity as maximum likelihood, and can be applied to any model for categorical responses. In addition, I develop a novel test that tests whether a given response can 
    
[^2]: TNDDR: 高效且双重鲁棒的COVID-19疫苗有效性估计在阴性测试设计下

    TNDDR: Efficient and doubly robust estimation of COVID-19 vaccine effectiveness under the test-negative design. (arXiv:2310.04578v1 [stat.ME])

    [http://arxiv.org/abs/2310.04578](http://arxiv.org/abs/2310.04578)

    我们提出了一种高效且双重鲁棒的估计器TNDDR，用于在阴性测试设计下估计COVID-19疫苗的有效性，可有效解决选择偏差问题，并结合机器学习技术进行辅助函数估计。

    

    尽管阴性测试设计（TND）常用于监测季节性流感疫苗有效性（VE），但最近已成为COVID-19疫苗监测的重要组成部分，但由于结果相关抽样，它容易受到选择偏差的影响。一些研究已经解决了TND下因果参数的可鉴别性和估计问题，但尚未研究非参数估计器在无混杂性假设下的效率边界。我们提出了一种称为TNDDR（TND双重鲁棒）的一步双重鲁棒和局部高效估计器,它利用样本分割，并可以结合机器学习技术来估计辅助函数。我们推导了结果边际期望的高效影响函数（EIF），探索了von Mises展开，并建立了TNDDR的n的平方根一致性、渐近正态性和双重鲁棒性的条件。

    While the test-negative design (TND), which is routinely used for monitoring seasonal flu vaccine effectiveness (VE), has recently become integral to COVID-19 vaccine surveillance, it is susceptible to selection bias due to outcome-dependent sampling. Some studies have addressed the identifiability and estimation of causal parameters under the TND, but efficiency bounds for nonparametric estimators of the target parameter under the unconfoundedness assumption have not yet been investigated. We propose a one-step doubly robust and locally efficient estimator called TNDDR (TND doubly robust), which utilizes sample splitting and can incorporate machine learning techniques to estimate the nuisance functions. We derive the efficient influence function (EIF) for the marginal expectation of the outcome under a vaccination intervention, explore the von Mises expansion, and establish the conditions for $\sqrt{n}-$consistency, asymptotic normality and double robustness of TNDDR. The proposed TND
    

