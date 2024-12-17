# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [E-backtesting.](http://arxiv.org/abs/2209.00991) | 本文提出了一种基于E-值和E-过程技术的ES预测无模型回测程序，可以自然地应用于许多其他风险度量和统计量。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: E-backtesting——一种ES预测的无模型回测程序

    E-backtesting. (arXiv:2209.00991v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2209.00991](http://arxiv.org/abs/2209.00991)

    本文提出了一种基于E-值和E-过程技术的ES预测无模型回测程序，可以自然地应用于许多其他风险度量和统计量。

    

    在最近的巴塞尔协议中，预期损失（ES）取代了价值损失（VaR）成为银行业市场风险的标准风险度量，使它成为金融监管中最重要的风险度量。风险建模实践中最具挑战性的任务之一是回测金融机构提供的ES预测。为了设计一种ES的无模型回测程序，我们利用了最近发展的E-值和E-过程技术。引入了无模型E-统计量来制定风险度量预测的E-过程，并利用鉴别函数的最新结果为VaR和ES的无模型E-统计量确定了独特的形式。对于给定的无模型E-统计量，研究了构建E-过程的最优方法。该方法可以自然地应用于许多其他风险度量和统计量。我们进行了大量的模拟研究和数据分析，以说明无模型方法的优势。

    In the recent Basel Accords, the Expected Shortfall (ES) replaces the Value-at-Risk (VaR) as the standard risk measure for market risk in the banking sector, making it the most important risk measure in financial regulation. One of the most challenging tasks in risk modeling practice is to backtest ES forecasts provided by financial institutions. To design a model-free backtesting procedure for ES, we make use of the recently developed techniques of e-values and e-processes. Model-free e-statistics are introduced to formulate e-processes for risk measure forecasts, and unique forms of model-free e-statistics for VaR and ES are characterized using recent results on identification functions. For a given model-free e-statistic, optimal ways of constructing the e-processes are studied. The proposed method can be naturally applied to many other risk measures and statistical quantities. We conduct extensive simulation studies and data analysis to illustrate the advantages of the model-free b
    

