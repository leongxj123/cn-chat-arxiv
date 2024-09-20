# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Targeting Relative Risk Heterogeneity with Causal Forests.](http://arxiv.org/abs/2309.15793) | 本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。 |
| [^2] | [Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits.](http://arxiv.org/abs/2306.06291) | 本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。 |

# 详细

[^1]: 用因果森林针对相对风险异质性进行目标化

    Targeting Relative Risk Heterogeneity with Causal Forests. (arXiv:2309.15793v1 [stat.ME])

    [http://arxiv.org/abs/2309.15793](http://arxiv.org/abs/2309.15793)

    本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。

    

    在临床试验分析中，治疗效应异质性（TEH）即种群中不同亚群的治疗效应的变异性是非常重要的。因果森林（Wager和Athey，2018）是解决这个问题的一种非常流行的方法，但像许多其他发现TEH的方法一样，它用于分离亚群的标准侧重于绝对风险的差异。这可能会削弱统计功效，掩盖了相对风险中的细微差别，而相对风险通常是临床关注的更合适的数量。在这项工作中，我们提出并实现了一种修改因果森林以针对相对风险的方法，使用基于广义线性模型（GLM）比较的新颖节点分割过程。我们在模拟和真实数据上展示了结果，表明相对风险的因果森林可以捕捉到其他未观察到的异质性源。

    Treatment effect heterogeneity (TEH), or variability in treatment effect for different subgroups within a population, is of significant interest in clinical trial analysis. Causal forests (Wager and Athey, 2018) is a highly popular method for this problem, but like many other methods for detecting TEH, its criterion for separating subgroups focuses on differences in absolute risk. This can dilute statistical power by masking nuance in the relative risk, which is often a more appropriate quantity of clinical interest. In this work, we propose and implement a methodology for modifying causal forests to target relative risk using a novel node-splitting procedure based on generalized linear model (GLM) comparison. We present results on simulated and real-world data that suggest relative risk causal forests can capture otherwise unobserved sources of heterogeneity.
    
[^2]: 最优异构协同线性回归和上下文臂研究

    Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits. (arXiv:2306.06291v1 [stat.ML])

    [http://arxiv.org/abs/2306.06291](http://arxiv.org/abs/2306.06291)

    本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。

    

    大型和复杂的数据集往往来自于几个可能是异构的来源。协同学习方法通过利用数据集之间的共性提高效率，同时考虑可能出现的差异。在这里，我们研究协同线性回归和上下文臂问题，其中每个实例的相关参数等于全局参数加上一个稀疏的实例特定术语。我们提出了一种名为MOLAR的新型二阶段估计器，它通过首先构建实例线性回归估计的逐项中位数，然后将实例特定估计值收缩到中位数附近来利用这种结构。与独立最小二乘估计相比，MOLAR提高了估计误差对数据维度的依赖性。然后，我们将MOLAR应用于开发用于稀疏异构协同上下文臂的方法，这些方法相比独立臂模型具有更好的遗憾保证。我们进一步证明了我们的贡献优于先前在文献中报道的算法。

    Large and complex datasets are often collected from several, possibly heterogeneous sources. Collaborative learning methods improve efficiency by leveraging commonalities across datasets while accounting for possible differences among them. Here we study collaborative linear regression and contextual bandits, where each instance's associated parameters are equal to a global parameter plus a sparse instance-specific term. We propose a novel two-stage estimator called MOLAR that leverages this structure by first constructing an entry-wise median of the instances' linear regression estimates, and then shrinking the instance-specific estimates towards the median. MOLAR improves the dependence of the estimation error on the data dimension, compared to independent least squares estimates. We then apply MOLAR to develop methods for sparsely heterogeneous collaborative contextual bandits, which lead to improved regret guarantees compared to independent bandit methods. We further show that our 
    

