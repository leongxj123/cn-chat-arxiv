# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^2] | [Targeting Relative Risk Heterogeneity with Causal Forests.](http://arxiv.org/abs/2309.15793) | 本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。 |
| [^3] | [The Fundamental Limits of Structure-Agnostic Functional Estimation.](http://arxiv.org/abs/2305.04116) | 一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。 |

# 详细

[^1]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^2]: 用因果森林针对相对风险异质性进行目标化

    Targeting Relative Risk Heterogeneity with Causal Forests. (arXiv:2309.15793v1 [stat.ME])

    [http://arxiv.org/abs/2309.15793](http://arxiv.org/abs/2309.15793)

    本研究提出了一种通过修改因果森林方法，以相对风险为目标，从而捕捉到治疗效应异质性的潜在来源。

    

    在临床试验分析中，治疗效应异质性（TEH）即种群中不同亚群的治疗效应的变异性是非常重要的。因果森林（Wager和Athey，2018）是解决这个问题的一种非常流行的方法，但像许多其他发现TEH的方法一样，它用于分离亚群的标准侧重于绝对风险的差异。这可能会削弱统计功效，掩盖了相对风险中的细微差别，而相对风险通常是临床关注的更合适的数量。在这项工作中，我们提出并实现了一种修改因果森林以针对相对风险的方法，使用基于广义线性模型（GLM）比较的新颖节点分割过程。我们在模拟和真实数据上展示了结果，表明相对风险的因果森林可以捕捉到其他未观察到的异质性源。

    Treatment effect heterogeneity (TEH), or variability in treatment effect for different subgroups within a population, is of significant interest in clinical trial analysis. Causal forests (Wager and Athey, 2018) is a highly popular method for this problem, but like many other methods for detecting TEH, its criterion for separating subgroups focuses on differences in absolute risk. This can dilute statistical power by masking nuance in the relative risk, which is often a more appropriate quantity of clinical interest. In this work, we propose and implement a methodology for modifying causal forests to target relative risk using a novel node-splitting procedure based on generalized linear model (GLM) comparison. We present results on simulated and real-world data that suggest relative risk causal forests can capture otherwise unobserved sources of heterogeneity.
    
[^3]: 结构无关函数估计的基本限制

    The Fundamental Limits of Structure-Agnostic Functional Estimation. (arXiv:2305.04116v1 [math.ST])

    [http://arxiv.org/abs/2305.04116](http://arxiv.org/abs/2305.04116)

    一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。

    

    近年来，许多因果推断和函数估计问题的发展都源于这样一个事实：在非常弱的条件下，经典的一步（一阶）去偏方法或它们较新的样本分割双机器学习方法可以比插补估计更好地工作。这些一阶校正以黑盒子方式改善插补估计值，因此经常与强大的现成估计方法一起使用。然而，当干扰函数生存在Holder型函数空间中时，这些一阶方法在最小二乘意义下被证明是次优的。这种一阶去偏的次优性促进了“高阶”去偏方法的发展。由此产生的估计量在某些情况下被证明是在Holder类型空间上最小化的，并且它们的分析与基础函数空间的性质密切相关。

    Many recent developments in causal inference, and functional estimation problems more generally, have been motivated by the fact that classical one-step (first-order) debiasing methods, or their more recent sample-split double machine-learning avatars, can outperform plugin estimators under surprisingly weak conditions. These first-order corrections improve on plugin estimators in a black-box fashion, and consequently are often used in conjunction with powerful off-the-shelf estimation methods. These first-order methods are however provably suboptimal in a minimax sense for functional estimation when the nuisance functions live in Holder-type function spaces. This suboptimality of first-order debiasing has motivated the development of "higher-order" debiasing methods. The resulting estimators are, in some cases, provably optimal over Holder-type spaces, but both the estimators which are minimax-optimal and their analyses are crucially tied to properties of the underlying function space
    

