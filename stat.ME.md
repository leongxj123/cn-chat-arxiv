# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Switchback Experiments under Geometric Mixing.](http://arxiv.org/abs/2209.00197) | 本文研究了几何混合条件下的切换试验性质，并发现在该设置下，标准的切换设计受到了延续偏差的影响，但是通过谨慎使用初始燃烧期可以显著改善情况，实现误差以更慢的速度衰减。 |
| [^2] | [Optimal Stratification of Survey Experiments.](http://arxiv.org/abs/2111.08157) | 本文研究了调查实验的最优分层设计，引入了细致分层设计方法，并提供了解决具有异质成本和固定预算的最优设计问题的简单启发式方法。这种设计能够减小治疗效应估计的方差，并提供了高效的一致估计和渐近确切推断方法。 |

# 详细

[^1]: 几何混合条件下的切换试验

    Switchback Experiments under Geometric Mixing. (arXiv:2209.00197v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2209.00197](http://arxiv.org/abs/2209.00197)

    本文研究了几何混合条件下的切换试验性质，并发现在该设置下，标准的切换设计受到了延续偏差的影响，但是通过谨慎使用初始燃烧期可以显著改善情况，实现误差以更慢的速度衰减。

    

    切换试验是一种通过反复对整个系统开启和关闭干预来测量治疗效果的实验设计。切换试验是克服单元间溢出效应的一种强大方法；然而，它们容易受到时间延续的偏差影响。本文研究在几何混合条件下的马尔可夫系统中的切换试验性质。我们发现，在这种情况下，标准的切换设计在延续偏差方面受到了较大的影响：它们的估计误差随着实验时间跨度$T$的增加而衰减为$T^{-1/3}$，而在没有延续效应的情况下，更快的$T^{-1/2}$衰减速度是可能的。然而，我们还展示了谨慎使用初始燃烧期可以大大改善情况，并且实现误差以$\log(T)^{1/2}T^{-1/2}$的速度衰减。我们的形式结果在实证评估中得到了验证。

    The switchback is an experimental design that measures treatment effects by repeatedly turning an intervention on and off for a whole system. Switchback experiments are a robust way to overcome cross-unit spillover effects; however, they are vulnerable to bias from temporal carryovers. In this paper, we consider properties of switchback experiments in Markovian systems that mix at a geometric rate. We find that, in this setting, standard switchback designs suffer considerably from carryover bias: Their estimation error decays as $T^{-1/3}$ in terms of the experiment horizon $T$, whereas in the absence of carryovers a faster rate of $T^{-1/2}$ would have been possible. We also show, however, that judicious use of burn-in periods can considerably improve the situation, and enables errors that decay as $\log(T)^{1/2}T^{-1/2}$. Our formal results are mirrored in an empirical evaluation.
    
[^2]: 调查实验的最优分层

    Optimal Stratification of Survey Experiments. (arXiv:2111.08157v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2111.08157](http://arxiv.org/abs/2111.08157)

    本文研究了调查实验的最优分层设计，引入了细致分层设计方法，并提供了解决具有异质成本和固定预算的最优设计问题的简单启发式方法。这种设计能够减小治疗效应估计的方差，并提供了高效的一致估计和渐近确切推断方法。

    

    本文研究了一个两阶段的实验模型，研究人员首先从一个符合条件的样本池中抽取代表性单位，然后将每个抽样单位分配到治疗组或对照组。为了实现平衡抽样和分配，我们引入了一种新的细致分层设计，将匹配对随机分配推广到概率p(x)不等于1/2的情况。我们展示了两阶段分层非参数地减小了治疗效应估计的方差。我们制定并解决了具有异质成本和固定预算的最优分层问题，提供了简单的启发式方法来确定最优设计。在具有试点数据的情况下，我们展示了实施这种设计的一致估计也是高效的，可以在预算约束条件下最小化渐近方差。我们还提供了新的渐近确切推断方法，使实验者能够充分利用分层抽样和分配带来的效率提高。应用于

    This paper studies a two-stage model of experimentation, where the researcher first samples representative units from an eligible pool, then assigns each sampled unit to treatment or control. To implement balanced sampling and assignment, we introduce a new family of finely stratified designs that generalize matched pairs randomization to propensities p(x) not equal to 1/2. We show that two-stage stratification nonparametrically dampens the variance of treatment effect estimation. We formulate and solve the optimal stratification problem with heterogeneous costs and fixed budget, providing simple heuristics for the optimal design. In settings with pilot data, we show that implementing a consistent estimate of this design is also efficient, minimizing asymptotic variance subject to the budget constraint. We also provide new asymptotically exact inference methods, allowing experimenters to fully exploit the efficiency gains from both stratified sampling and assignment. An application to 
    

