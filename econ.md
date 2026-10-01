# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bounds on Average Effects in Discrete Choice Panel Data Models.](http://arxiv.org/abs/2309.09299) | 本文提出了一种在离散选择面板数据模型中对平均效应进行外界限估计的方法，无论协变量是离散还是连续，都可以在相对较大的样本中轻松获得，并提供了对识别集的渐近有效置信区间。 |
| [^2] | [Constrained Classification and Policy Learning.](http://arxiv.org/abs/2106.12886) | 研究了受限分类和策略学习中替代损失程序的一致性和适用性。 |

# 详细

[^1]: 离散选择面板数据模型中平均效应的界限

    Bounds on Average Effects in Discrete Choice Panel Data Models. (arXiv:2309.09299v1 [econ.EM])

    [http://arxiv.org/abs/2309.09299](http://arxiv.org/abs/2309.09299)

    本文提出了一种在离散选择面板数据模型中对平均效应进行外界限估计的方法，无论协变量是离散还是连续，都可以在相对较大的样本中轻松获得，并提供了对识别集的渐近有效置信区间。

    

    在具有个体特定固定效应的离散选择面板数据模型中，平均效应通常只在短期面板中部分识别。尽管可以对识别集进行一致估计，但通常需要非常大的样本量，特别是当观测协变量的支持点数量很大时，例如协变量是连续的情况。在本文中，我们提出了对平均效应的识别集进行外界限估计的方法。我们的界限易于构建，收敛速度为参数速度，并且在样本相对较大的情况下，无论协变量是离散还是连续，都很容易获取。我们还提供了对识别集的渐近有效置信区间。模拟研究证实我们的方法在有限样本中表现良好且具有信息价值。我们还考虑了劳动力参与的应用。

    Average effects in discrete choice panel data models with individual-specific fixed effects are generally only partially identified in short panels. While consistent estimation of the identified set is possible, it generally requires very large sample sizes, especially when the number of support points of the observed covariates is large, such as when the covariates are continuous. In this paper, we propose estimating outer bounds on the identified set of average effects. Our bounds are easy to construct, converge at the parametric rate, and are computationally simple to obtain even in moderately large samples, independent of whether the covariates are discrete or continuous. We also provide asymptotically valid confidence intervals on the identified set. Simulation studies confirm that our approach works well and is informative in finite samples. We also consider an application to labor force participation.
    
[^2]: 受限分类和策略学习

    Constrained Classification and Policy Learning. (arXiv:2106.12886v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2106.12886](http://arxiv.org/abs/2106.12886)

    研究了受限分类和策略学习中替代损失程序的一致性和适用性。

    

    现代机器学习方法对于分类问题使用了一些替代损失技术，如AdaBoost、支持向量机和深度神经网络，以绕过最小化经验分类风险的计算复杂性。这些技术在因果策略学习问题中也很有用，因为个性化治疗规则的估计可以被视为一种加权（成本敏感）分类问题。Zhang（2004年）和Bartlett等人（2006年）研究的替代损失方法的一致性关键依赖于正确规范的假设，即指定的分类器集合足够丰富，包含一个最佳分类器。然而，当分类器集合受到可解释性或公平性的限制时，这个假设较不可靠，这导致在这种次佳情景下替代损失方法的适用性未知。本文研究了在受限类集合条件下的替代损失程序的一致性。

    Modern machine learning approaches to classification, including AdaBoost, support vector machines, and deep neural networks, utilize surrogate loss techniques to circumvent the computational complexity of minimizing empirical classification risk. These techniques are also useful for causal policy learning problems, since estimation of individualized treatment rules can be cast as a weighted (cost-sensitive) classification problem. Consistency of the surrogate loss approaches studied in Zhang (2004) and Bartlett et al. (2006) crucially relies on the assumption of correct specification, meaning that the specified set of classifiers is rich enough to contain a first-best classifier. This assumption is, however, less credible when the set of classifiers is constrained by interpretability or fairness, leaving the applicability of surrogate loss based algorithms unknown in such second-best scenarios. This paper studies consistency of surrogate loss procedures under a constrained set of class
    

