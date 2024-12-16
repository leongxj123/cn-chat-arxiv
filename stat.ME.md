# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semiparametric Inference for Regression-Discontinuity Designs](https://arxiv.org/abs/2403.05803) | 本文提出了一种用于在回归断点设计中估计治疗效应的半参数方法，通过将RDDs中的治疗效应识别概念化为部分线性建模问题，并利用P-样条方法逼近非参数函数，开发了推断治疗效应的程序。 |
| [^2] | [Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits.](http://arxiv.org/abs/2306.06291) | 本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。 |

# 详细

[^1]: 回归断点设计的半参数推断

    Semiparametric Inference for Regression-Discontinuity Designs

    [https://arxiv.org/abs/2403.05803](https://arxiv.org/abs/2403.05803)

    本文提出了一种用于在回归断点设计中估计治疗效应的半参数方法，通过将RDDs中的治疗效应识别概念化为部分线性建模问题，并利用P-样条方法逼近非参数函数，开发了推断治疗效应的程序。

    

    在回归断点设计（RDDs）中，治疗效应通常使用局部回归方法进行估计。然而，全局近似方法通常被认为效率低下。本文提出了一个专门用于在RDDs中估计治疗效应的半参数框架。我们的全局方法将RDDs中的治疗效应识别概念化为部分线性建模问题，其中线性部分捕捉治疗效应。此外，我们利用P-样条方法来逼近非参数函数，并开发了推断在这个框架内的治疗效应的程序。我们通过蒙特卡洛模拟表明，所提出的方法在各种场景下表现良好。此外，我们通过真实数据集的例证，说明我们的全局方法可能导致更可靠的推断。

    arXiv:2403.05803v1 Announce Type: new  Abstract: Treatment effects in regression discontinuity designs (RDDs) are often estimated using local regression methods. However, global approximation methods are generally deemed inefficient. In this paper, we propose a semiparametric framework tailored for estimating treatment effects in RDDs. Our global approach conceptualizes the identification of treatment effects within RDDs as a partially linear modeling problem, with the linear component capturing the treatment effect. Furthermore, we utilize the P-spline method to approximate the nonparametric function and develop procedures for inferring treatment effects within this framework. We demonstrate through Monte Carlo simulations that the proposed method performs well across various scenarios. Furthermore, we illustrate using real-world datasets that our global approach may result in more reliable inference.
    
[^2]: 最优异构协同线性回归和上下文臂研究

    Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits. (arXiv:2306.06291v1 [stat.ML])

    [http://arxiv.org/abs/2306.06291](http://arxiv.org/abs/2306.06291)

    本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。

    

    大型和复杂的数据集往往来自于几个可能是异构的来源。协同学习方法通过利用数据集之间的共性提高效率，同时考虑可能出现的差异。在这里，我们研究协同线性回归和上下文臂问题，其中每个实例的相关参数等于全局参数加上一个稀疏的实例特定术语。我们提出了一种名为MOLAR的新型二阶段估计器，它通过首先构建实例线性回归估计的逐项中位数，然后将实例特定估计值收缩到中位数附近来利用这种结构。与独立最小二乘估计相比，MOLAR提高了估计误差对数据维度的依赖性。然后，我们将MOLAR应用于开发用于稀疏异构协同上下文臂的方法，这些方法相比独立臂模型具有更好的遗憾保证。我们进一步证明了我们的贡献优于先前在文献中报道的算法。

    Large and complex datasets are often collected from several, possibly heterogeneous sources. Collaborative learning methods improve efficiency by leveraging commonalities across datasets while accounting for possible differences among them. Here we study collaborative linear regression and contextual bandits, where each instance's associated parameters are equal to a global parameter plus a sparse instance-specific term. We propose a novel two-stage estimator called MOLAR that leverages this structure by first constructing an entry-wise median of the instances' linear regression estimates, and then shrinking the instance-specific estimates towards the median. MOLAR improves the dependence of the estimation error on the data dimension, compared to independent least squares estimates. We then apply MOLAR to develop methods for sparsely heterogeneous collaborative contextual bandits, which lead to improved regret guarantees compared to independent bandit methods. We further show that our 
    

