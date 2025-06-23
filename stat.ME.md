# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning.](http://arxiv.org/abs/2401.03756) | 该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。 |
| [^2] | [Conformal prediction for frequency-severity modeling.](http://arxiv.org/abs/2307.13124) | 这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。 |
| [^3] | [Assessing Omitted Variable Bias when the Controls are Endogenous.](http://arxiv.org/abs/2206.02303) | 该论文提出了一种新的敏感性分析方法，避免了传统方法中常常被认为是强假设和不可行的假设，允许省略的变量与包含的控制变量相关，并允许研究人员校准灵敏度。 |

# 详细

[^1]: 上下文固定预算的最佳臂识别：适应性实验设计与策略学习

    Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning. (arXiv:2401.03756v1 [cs.LG])

    [http://arxiv.org/abs/2401.03756](http://arxiv.org/abs/2401.03756)

    该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。

    

    个性化治疗推荐是基于证据的决策中的关键任务。在这项研究中，我们将这个任务作为一个带有上下文信息的固定预算最佳臂识别（Best Arm Identification, BAI）问题来进行建模。在这个设置中，我们考虑了一个给定多个治疗臂的自适应试验。在每一轮中，决策者观察一个刻画实验单位的上下文（协变量），并将该单位分配给其中一个治疗臂。在实验结束时，决策者推荐一个在给定上下文条件下预计产生最高期望结果的治疗臂（最佳治疗臂）。该决策的有效性通过最坏情况下的期望简单遗憾（策略遗憾）来衡量，该遗憾表示在给定上下文条件下，最佳治疗臂和推荐治疗臂的条件期望结果之间的最大差异。我们的初始步骤是推导最坏情况下期望简单遗憾的渐近下界，该下界还暗示着解决该问题的一些思路。

    Individualized treatment recommendation is a crucial task in evidence-based decision-making. In this study, we formulate this task as a fixed-budget best arm identification (BAI) problem with contextual information. In this setting, we consider an adaptive experiment given multiple treatment arms. At each round, a decision-maker observes a context (covariate) that characterizes an experimental unit and assigns the unit to one of the treatment arms. At the end of the experiment, the decision-maker recommends a treatment arm estimated to yield the highest expected outcome conditioned on a context (best treatment arm). The effectiveness of this decision is measured in terms of the worst-case expected simple regret (policy regret), which represents the largest difference between the conditional expected outcomes of the best and recommended treatment arms given a context. Our initial step is to derive asymptotic lower bounds for the worst-case expected simple regret, which also implies idea
    
[^2]: 频率-严重性建模的符合性预测

    Conformal prediction for frequency-severity modeling. (arXiv:2307.13124v1 [stat.ME])

    [http://arxiv.org/abs/2307.13124](http://arxiv.org/abs/2307.13124)

    这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。

    

    我们提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，将分割符合性预测技术扩展到两阶段频率-严重性建模领域。通过模拟和真实数据集展示了该框架的有效性。当基础严重性模型是随机森林时，我们扩展了两阶段分割符合性预测过程，展示了如何利用袋外机制消除校准集的需要，并实现具有自适应宽度的预测区间的生成。

    We present a nonparametric model-agnostic framework for building prediction intervals of insurance claims, with finite sample statistical guarantees, extending the technique of split conformal prediction to the domain of two-stage frequency-severity modeling. The effectiveness of the framework is showcased with simulated and real datasets. When the underlying severity model is a random forest, we extend the two-stage split conformal prediction procedure, showing how the out-of-bag mechanism can be leveraged to eliminate the need for a calibration set and to enable the production of prediction intervals with adaptive width.
    
[^3]: 当控制变量存在内生性时，评估省略变量偏误

    Assessing Omitted Variable Bias when the Controls are Endogenous. (arXiv:2206.02303v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.02303](http://arxiv.org/abs/2206.02303)

    该论文提出了一种新的敏感性分析方法，避免了传统方法中常常被认为是强假设和不可行的假设，允许省略的变量与包含的控制变量相关，并允许研究人员校准灵敏度。

    

    省略变量是导致因果效应识别受到最大威胁的因素之一。包括Oster（2019）在内的几种广泛使用的方法通过将可观测选择测量与不可观测选择测量进行比较来评估省略变量对经验结论的影响。这些方法要么（1）假设省略的变量与包括的控制变量不相关，这个假设常常被认为是强假设和不可行的，要么（2）使用残差法来避免这个假设。在我们的第一项贡献中，我们开发了一个框架，用于客观地比较敏感度参数。我们利用这个框架正式证明残差化方法通常会导致有关鲁棒性的错误结论。在我们的第二项贡献中，我们提出了一种新的敏感性分析方法，避免了这个批评，允许省略的变量与包含的控制变量相关，并允许研究人员校准灵敏度。

    Omitted variables are one of the most important threats to the identification of causal effects. Several widely used approaches, including Oster (2019), assess the impact of omitted variables on empirical conclusions by comparing measures of selection on observables with measures of selection on unobservables. These approaches either (1) assume the omitted variables are uncorrelated with the included controls, an assumption that is often considered strong and implausible, or (2) use a method called residualization to avoid this assumption. In our first contribution, we develop a framework for objectively comparing sensitivity parameters. We use this framework to formally prove that the residualization method generally leads to incorrect conclusions about robustness. In our second contribution, we then provide a new approach to sensitivity analysis that avoids this critique, allows the omitted variables to be correlated with the included controls, and lets researchers calibrate sensitiv
    

