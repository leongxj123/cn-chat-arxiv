# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective.](http://arxiv.org/abs/2304.13836) | 本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。 |
| [^2] | [Optimal Stratification of Survey Experiments.](http://arxiv.org/abs/2111.08157) | 本文研究了调查实验的最优分层设计，引入了细致分层设计方法，并提供了解决具有异质成本和固定预算的最优设计问题的简单启发式方法。这种设计能够减小治疗效应估计的方差，并提供了高效的一致估计和渐近确切推断方法。 |

# 详细

[^1]: 论RemOve-And-Retrain的陷阱：数据处理不等式的视角

    On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective. (arXiv:2304.13836v1 [cs.LG])

    [http://arxiv.org/abs/2304.13836](http://arxiv.org/abs/2304.13836)

    本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。

    

    本文评估了RemOve-And-Retrain（ROAR）协议的可靠性，该协议用于测量特征重要性估计的性能。我们从理论背景和实证实验中发现，具有较少有关决策功能的信息的属性在ROAR基准测试中表现更好，与ROAR的原始目的相矛盾。这种现象也出现在最近提出的变体RemOve-And-Debias（ROAD）中，我们提出了ROAR归因度量中毛糙度偏差的一致趋势。我们的结果提醒人们不要盲目依赖ROAR的性能评估指标。

    This paper assesses the reliability of the RemOve-And-Retrain (ROAR) protocol, which is used to measure the performance of feature importance estimates. Our findings from the theoretical background and empirical experiments indicate that attributions that possess less information about the decision function can perform better in ROAR benchmarks, conflicting with the original purpose of ROAR. This phenomenon is also observed in the recently proposed variant RemOve-And-Debias (ROAD), and we propose a consistent trend of blurriness bias in ROAR attribution metrics. Our results caution against uncritical reliance on ROAR metrics.
    
[^2]: 调查实验的最优分层

    Optimal Stratification of Survey Experiments. (arXiv:2111.08157v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2111.08157](http://arxiv.org/abs/2111.08157)

    本文研究了调查实验的最优分层设计，引入了细致分层设计方法，并提供了解决具有异质成本和固定预算的最优设计问题的简单启发式方法。这种设计能够减小治疗效应估计的方差，并提供了高效的一致估计和渐近确切推断方法。

    

    本文研究了一个两阶段的实验模型，研究人员首先从一个符合条件的样本池中抽取代表性单位，然后将每个抽样单位分配到治疗组或对照组。为了实现平衡抽样和分配，我们引入了一种新的细致分层设计，将匹配对随机分配推广到概率p(x)不等于1/2的情况。我们展示了两阶段分层非参数地减小了治疗效应估计的方差。我们制定并解决了具有异质成本和固定预算的最优分层问题，提供了简单的启发式方法来确定最优设计。在具有试点数据的情况下，我们展示了实施这种设计的一致估计也是高效的，可以在预算约束条件下最小化渐近方差。我们还提供了新的渐近确切推断方法，使实验者能够充分利用分层抽样和分配带来的效率提高。应用于

    This paper studies a two-stage model of experimentation, where the researcher first samples representative units from an eligible pool, then assigns each sampled unit to treatment or control. To implement balanced sampling and assignment, we introduce a new family of finely stratified designs that generalize matched pairs randomization to propensities p(x) not equal to 1/2. We show that two-stage stratification nonparametrically dampens the variance of treatment effect estimation. We formulate and solve the optimal stratification problem with heterogeneous costs and fixed budget, providing simple heuristics for the optimal design. In settings with pilot data, we show that implementing a consistent estimate of this design is also efficient, minimizing asymptotic variance subject to the budget constraint. We also provide new asymptotically exact inference methods, allowing experimenters to fully exploit the efficiency gains from both stratified sampling and assignment. An application to 
    

