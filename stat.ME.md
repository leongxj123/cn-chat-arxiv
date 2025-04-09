# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference with Mondrian Random Forests.](http://arxiv.org/abs/2310.09702) | 本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。 |
| [^2] | [Accelerating Generalized Random Forests with Fixed-Point Trees.](http://arxiv.org/abs/2306.11908) | 本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。 |

# 详细

[^1]: 带有Mondrian随机森林的推理

    Inference with Mondrian Random Forests. (arXiv:2310.09702v1 [math.ST])

    [http://arxiv.org/abs/2310.09702](http://arxiv.org/abs/2310.09702)

    本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。

    

    随机森林是一种常用的分类和回归方法，在最近几年中提出了许多不同的变体。一个有趣的例子是Mondrian随机森林，其中底层树是根据Mondrian过程构建的。在本文中，我们给出了Mondrian随机森林在回归设置下的估计的中心极限定理。当与偏差表征和一致方差估计器相结合时，这允许进行渐近有效的统计推断，如构建置信区间，对未知的回归函数进行推断。我们还提供了一种去偏过程，用于Mondrian随机森林，使其能够在适当的参数调整下实现$\beta$-H\"older回归函数的最小极大估计速率，对于所有的$\beta$和任意维度。

    Random forests are popular methods for classification and regression, and many different variants have been proposed in recent years. One interesting example is the Mondrian random forest, in which the underlying trees are constructed according to a Mondrian process. In this paper we give a central limit theorem for the estimates made by a Mondrian random forest in the regression setting. When combined with a bias characterization and a consistent variance estimator, this allows one to perform asymptotically valid statistical inference, such as constructing confidence intervals, on the unknown regression function. We also provide a debiasing procedure for Mondrian random forests which allows them to achieve minimax-optimal estimation rates with $\beta$-H\"older regression functions, for all $\beta$ and in arbitrary dimension, assuming appropriate parameter tuning.
    
[^2]: 基于定点树的广义随机森林加速

    Accelerating Generalized Random Forests with Fixed-Point Trees. (arXiv:2306.11908v1 [stat.ML])

    [http://arxiv.org/abs/2306.11908](http://arxiv.org/abs/2306.11908)

    本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。

    

    广义随机森林建立在传统随机森林的基础上，通过将其作为自适应核加权算法来构建估算器，并通过基于梯度的树生长过程来实现。我们提出了一种新的树生长规则，基于定点迭代近似表示梯度近似，实现了无梯度优化，并为此开发了渐近理论。这有效地节省了时间，尤其是在目标量的维度适中时。

    Generalized random forests arXiv:1610.01271 build upon the well-established success of conventional forests (Breiman, 2001) to offer a flexible and powerful non-parametric method for estimating local solutions of heterogeneous estimating equations. Estimators are constructed by leveraging random forests as an adaptive kernel weighting algorithm and implemented through a gradient-based tree-growing procedure. By expressing this gradient-based approximation as being induced from a single Newton-Raphson root-finding iteration, and drawing upon the connection between estimating equations and fixed-point problems arXiv:2110.11074, we propose a new tree-growing rule for generalized random forests induced from a fixed-point iteration type of approximation, enabling gradient-free optimization, and yielding substantial time savings for tasks involving even modest dimensionality of the target quantity (e.g. multiple/multi-level treatment effects). We develop an asymptotic theory for estimators o
    

