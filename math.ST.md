# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation](https://arxiv.org/abs/2402.14264) | 采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性 |
| [^2] | [The Fundamental Limits of Structure-Agnostic Functional Estimation.](http://arxiv.org/abs/2305.04116) | 一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。 |

# 详细

[^1]: 双稳健学习在处理效应估计中的结构不可知性最优性

    Structure-agnostic Optimality of Doubly Robust Learning for Treatment Effect Estimation

    [https://arxiv.org/abs/2402.14264](https://arxiv.org/abs/2402.14264)

    采用结构不可知的统计下界框架，证明了双稳健估计器在平均处理效应（ATE）和平均处理效应方面的统计最优性

    

    平均处理效应估计是因果推断中最核心的问题，应用广泛。虽然文献中提出了许多估计策略，最近还纳入了通用的机器学习估计器，但这些方法的统计最优性仍然是一个开放的研究领域。本文采用最近引入的统计下界结构不可知框架，该框架对干扰函数没有结构性质假设，除了访问黑盒估计器以达到小误差；当只愿意考虑使用非参数回归和分类神谕作为黑盒子过程的估计策略时，这一点尤其吸引人。在这个框架内，我们证明了双稳健估计器对于平均处理效应（ATE）和平均处理效应的统计最优性。

    arXiv:2402.14264v1 Announce Type: cross  Abstract: Average treatment effect estimation is the most central problem in causal inference with application to numerous disciplines. While many estimation strategies have been proposed in the literature, recently also incorporating generic machine learning estimators, the statistical optimality of these methods has still remained an open area of investigation. In this paper, we adopt the recently introduced structure-agnostic framework of statistical lower bounds, which poses no structural properties on the nuisance functions other than access to black-box estimators that attain small errors; which is particularly appealing when one is only willing to consider estimation strategies that use non-parametric regression and classification oracles as a black-box sub-process. Within this framework, we prove the statistical optimality of the celebrated and widely used doubly robust estimators for both the Average Treatment Effect (ATE) and the Avera
    
[^2]: 结构无关函数估计的基本限制

    The Fundamental Limits of Structure-Agnostic Functional Estimation. (arXiv:2305.04116v1 [math.ST])

    [http://arxiv.org/abs/2305.04116](http://arxiv.org/abs/2305.04116)

    一阶去偏方法在最小二乘意义下在干扰函数生存在特定函数空间时被证明是次优的，这促进了“高阶”去偏方法的发展。

    

    近年来，许多因果推断和函数估计问题的发展都源于这样一个事实：在非常弱的条件下，经典的一步（一阶）去偏方法或它们较新的样本分割双机器学习方法可以比插补估计更好地工作。这些一阶校正以黑盒子方式改善插补估计值，因此经常与强大的现成估计方法一起使用。然而，当干扰函数生存在Holder型函数空间中时，这些一阶方法在最小二乘意义下被证明是次优的。这种一阶去偏的次优性促进了“高阶”去偏方法的发展。由此产生的估计量在某些情况下被证明是在Holder类型空间上最小化的，并且它们的分析与基础函数空间的性质密切相关。

    Many recent developments in causal inference, and functional estimation problems more generally, have been motivated by the fact that classical one-step (first-order) debiasing methods, or their more recent sample-split double machine-learning avatars, can outperform plugin estimators under surprisingly weak conditions. These first-order corrections improve on plugin estimators in a black-box fashion, and consequently are often used in conjunction with powerful off-the-shelf estimation methods. These first-order methods are however provably suboptimal in a minimax sense for functional estimation when the nuisance functions live in Holder-type function spaces. This suboptimality of first-order debiasing has motivated the development of "higher-order" debiasing methods. The resulting estimators are, in some cases, provably optimal over Holder-type spaces, but both the estimators which are minimax-optimal and their analyses are crucially tied to properties of the underlying function space
    

