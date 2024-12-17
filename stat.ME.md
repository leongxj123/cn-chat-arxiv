# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [Estimating Causal Effects of Discrete and Continuous Treatments with Binary Instruments](https://arxiv.org/abs/2403.05850) | 提出了一个利用二元工具的工具变量框架，用于识别和估计离散和连续处理的平均效应和分位数效应，基于Copula不变性的识别假设，构造性识别了整个人群及其他子人群的处理效应，并提出了基于分布回归的直接半参数估计过程。 |
| [^3] | [Nonparametric Causal Decomposition of Group Disparities.](http://arxiv.org/abs/2306.16591) | 本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。 |
| [^4] | [E-backtesting.](http://arxiv.org/abs/2209.00991) | 本文提出了一种基于E-值和E-过程技术的ES预测无模型回测程序，可以自然地应用于许多其他风险度量和统计量。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: 用二元工具估计离散和连续处理的因果效应

    Estimating Causal Effects of Discrete and Continuous Treatments with Binary Instruments

    [https://arxiv.org/abs/2403.05850](https://arxiv.org/abs/2403.05850)

    提出了一个利用二元工具的工具变量框架，用于识别和估计离散和连续处理的平均效应和分位数效应，基于Copula不变性的识别假设，构造性识别了整个人群及其他子人群的处理效应，并提出了基于分布回归的直接半参数估计过程。

    

    我们提出了一个利用二元工具的工具变量框架，用于识别和估计离散和连续处理的平均效应和分位数效应。我们方法的基础是潜在结果和决定处理分配的不可观测变量的联合分布的局部Copula表示。这种表示使我们能够引入一个所谓的Copula不变性的识别假设，该假设限制了Copula关于处理倾向的局部依赖。我们展示Copula不变性识别了整个人群以及其他亚人群（如接受处理者）的处理效应。识别结果是构造性的，并导致基于分布回归的直接半参数估计过程。对睡眠对幸福感的影响的应用揭示了有趣的异质性模式。

    arXiv:2403.05850v1 Announce Type: new  Abstract: We propose an instrumental variable framework for identifying and estimating average and quantile effects of discrete and continuous treatments with binary instruments. The basis of our approach is a local copula representation of the joint distribution of the potential outcomes and unobservables determining treatment assignment. This representation allows us to introduce an identifying assumption, so-called copula invariance, that restricts the local dependence of the copula with respect to the treatment propensity. We show that copula invariance identifies treatment effects for the entire population and other subpopulations such as the treated. The identification results are constructive and lead to straightforward semiparametric estimation procedures based on distribution regression. An application to the effect of sleep on well-being uncovers interesting patterns of heterogeneity.
    
[^3]: 非参数因果分解组差异

    Nonparametric Causal Decomposition of Group Disparities. (arXiv:2306.16591v1 [stat.ME])

    [http://arxiv.org/abs/2306.16591](http://arxiv.org/abs/2306.16591)

    本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。

    

    我们提出了一个因果框架来将结果中的组差异分解为中间处理变量。我们的框架捕捉了基线潜在结果、处理前沿、平均处理效应和处理选择的组差异的贡献。这个框架以反事实的方式进行了数学表达，并且能够方便地指导政策干预。特别是，针对不同的处理选择进行的分解部分是特别新颖的，揭示了一种解释和改善差异的新机制。这个框架以因果术语重新定义了经典的Kitagawa-Blinder-Oaxaca分解，通过解释组差异而不是组效应来补充了因果中介分析，并解决了近期随机等化分解的概念困难。我们还提供了一个条件分解，允许研究人员在定义评估和相应的干预措施时纳入协变量。

    We propose a causal framework for decomposing a group disparity in an outcome in terms of an intermediate treatment variable. Our framework captures the contributions of group differences in baseline potential outcome, treatment prevalence, average treatment effect, and selection into treatment. This framework is counterfactually formulated and readily informs policy interventions. The decomposition component for differential selection into treatment is particularly novel, revealing a new mechanism for explaining and ameliorating disparities. This framework reformulates the classic Kitagawa-Blinder-Oaxaca decomposition in causal terms, supplements causal mediation analysis by explaining group disparities instead of group effects, and resolves conceptual difficulties of recent random equalization decompositions. We also provide a conditional decomposition that allows researchers to incorporate covariates in defining the estimands and corresponding interventions. We develop nonparametric
    
[^4]: E-backtesting——一种ES预测的无模型回测程序

    E-backtesting. (arXiv:2209.00991v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2209.00991](http://arxiv.org/abs/2209.00991)

    本文提出了一种基于E-值和E-过程技术的ES预测无模型回测程序，可以自然地应用于许多其他风险度量和统计量。

    

    在最近的巴塞尔协议中，预期损失（ES）取代了价值损失（VaR）成为银行业市场风险的标准风险度量，使它成为金融监管中最重要的风险度量。风险建模实践中最具挑战性的任务之一是回测金融机构提供的ES预测。为了设计一种ES的无模型回测程序，我们利用了最近发展的E-值和E-过程技术。引入了无模型E-统计量来制定风险度量预测的E-过程，并利用鉴别函数的最新结果为VaR和ES的无模型E-统计量确定了独特的形式。对于给定的无模型E-统计量，研究了构建E-过程的最优方法。该方法可以自然地应用于许多其他风险度量和统计量。我们进行了大量的模拟研究和数据分析，以说明无模型方法的优势。

    In the recent Basel Accords, the Expected Shortfall (ES) replaces the Value-at-Risk (VaR) as the standard risk measure for market risk in the banking sector, making it the most important risk measure in financial regulation. One of the most challenging tasks in risk modeling practice is to backtest ES forecasts provided by financial institutions. To design a model-free backtesting procedure for ES, we make use of the recently developed techniques of e-values and e-processes. Model-free e-statistics are introduced to formulate e-processes for risk measure forecasts, and unique forms of model-free e-statistics for VaR and ES are characterized using recent results on identification functions. For a given model-free e-statistic, optimal ways of constructing the e-processes are studied. The proposed method can be naturally applied to many other risk measures and statistical quantities. We conduct extensive simulation studies and data analysis to illustrate the advantages of the model-free b
    

