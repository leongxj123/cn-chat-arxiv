# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding](https://arxiv.org/abs/2402.02306) | 本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。 |
| [^2] | [Causal Rule Learning: Enhancing the Understanding of Heterogeneous Treatment Effect via Weighted Causal Rules.](http://arxiv.org/abs/2310.06746) | 通过因果规则学习，我们可以利用加权因果规则来估计和加强对异质治疗效应的理解。 |

# 详细

[^1]: 弹性贝叶斯g形式在具有时变混杂的因果生存分析中的应用

    A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding

    [https://arxiv.org/abs/2402.02306](https://arxiv.org/abs/2402.02306)

    本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。

    

    在具有时间至事件结果的纵向观察性研究中，因果分析的常见目标是在研究群体中估计在假设干预情景下的因果生存曲线。g形式是这种分析的一个特别有用的工具。为了增强传统的参数化g形式方法，我们开发了一种更灵活的贝叶斯g形式估计器。该估计器同时支持纵向预测和因果推断。它在模拟时变生成组件的建模中引入了贝叶斯附加回归树，旨在减轻由于模型错误规范造成的偏差。具体而言，我们引入了一类更通用的离散生存数据g形式。这些公式可以引入纵向平衡分数，这在处理越来越多的时变混杂因素时是一种有效的降维方法。

    In longitudinal observational studies with a time-to-event outcome, a common objective in causal analysis is to estimate the causal survival curve under hypothetical intervention scenarios within the study cohort. The g-formula is a particularly useful tool for this analysis. To enhance the traditional parametric g-formula approach, we developed a more adaptable Bayesian g-formula estimator. This estimator facilitates both longitudinal predictive and causal inference. It incorporates Bayesian additive regression trees in the modeling of the time-evolving generative components, aiming to mitigate bias due to model misspecification. Specifically, we introduce a more general class of g-formulas for discrete survival data. These formulas can incorporate the longitudinal balancing scores, which serve as an effective method for dimension reduction and are vital when dealing with an expanding array of time-varying confounders. The minimum sufficient formulation of these longitudinal balancing
    
[^2]: 因果规则学习：通过加权因果规则增强对异质治疗效应的理解

    Causal Rule Learning: Enhancing the Understanding of Heterogeneous Treatment Effect via Weighted Causal Rules. (arXiv:2310.06746v1 [cs.LG])

    [http://arxiv.org/abs/2310.06746](http://arxiv.org/abs/2310.06746)

    通过因果规则学习，我们可以利用加权因果规则来估计和加强对异质治疗效应的理解。

    

    解释性是利用机器学习方法估计异质治疗效应时的关键问题，特别是对于医疗应用来说，常常需要做出高风险决策。受到解释性的预测性、描述性、相关性框架的启发，我们提出了因果规则学习，该方法通过找到描述潜在子群的精细因果规则集来估计和增强我们对异质治疗效应的理解。因果规则学习包括三个阶段：规则发现、规则选择和规则分析。在规则发现阶段，我们利用因果森林生成一组具有相应子群平均治疗效应的因果规则池。选择阶段使用D-学习方法从这些规则中选择子集，将个体水平的治疗效应作为子群水平效应的线性组合进行解构。这有助于回答之前文献忽视的问题：如果一个个体同时属于多个不同的治疗子群，会怎么样呢？

    Interpretability is a key concern in estimating heterogeneous treatment effects using machine learning methods, especially for healthcare applications where high-stake decisions are often made. Inspired by the Predictive, Descriptive, Relevant framework of interpretability, we propose causal rule learning which finds a refined set of causal rules characterizing potential subgroups to estimate and enhance our understanding of heterogeneous treatment effects. Causal rule learning involves three phases: rule discovery, rule selection, and rule analysis. In the rule discovery phase, we utilize a causal forest to generate a pool of causal rules with corresponding subgroup average treatment effects. The selection phase then employs a D-learning method to select a subset of these rules to deconstruct individual-level treatment effects as a linear combination of the subgroup-level effects. This helps to answer an ignored question by previous literature: what if an individual simultaneously bel
    

