# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrated path stability selection](https://arxiv.org/abs/2403.15877) | 该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。 |
| [^2] | [Inference with Mondrian Random Forests.](http://arxiv.org/abs/2310.09702) | 本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。 |

# 详细

[^1]: 集成路径稳定选择

    Integrated path stability selection

    [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)

    该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。

    

    稳定选择是一种广泛用于改善特征选择算法性能的方法。然而，已发现稳定选择过于保守，导致灵敏度较低。此外，对期望的假阳性数量的理论界限E(FP)相对较松，难以知道实践中会有多少假阳性。在本文中，我们提出一种基于集成稳定路径而非最大化稳定路径的新方法。这产生了对E(FP)更紧密的界限，导致实践中具有更高灵敏度的特征选择标准，并且在与目标E(FP)匹配方面更好地校准。我们提出的方法与原始稳定选择算法需要相同数量的计算，且仅需要用户指定一个输入参数，即E(FP)的目标值。我们提供了性能的理论界限。

    arXiv:2403.15877v1 Announce Type: cross  Abstract: Stability selection is a widely used method for improving the performance of feature selection algorithms. However, stability selection has been found to be highly conservative, resulting in low sensitivity. Further, the theoretical bound on the expected number of false positives, E(FP), is relatively loose, making it difficult to know how many false positives to expect in practice. In this paper, we introduce a novel method for stability selection based on integrating the stability paths rather than maximizing over them. This yields a tighter bound on E(FP), resulting in a feature selection criterion that has higher sensitivity in practice and is better calibrated in terms of matching the target E(FP). Our proposed method requires the same amount of computation as the original stability selection algorithm, and only requires the user to specify one input parameter, a target value for E(FP). We provide theoretical bounds on performance
    
[^2]: 带有Mondrian随机森林的推理

    Inference with Mondrian Random Forests. (arXiv:2310.09702v1 [math.ST])

    [http://arxiv.org/abs/2310.09702](http://arxiv.org/abs/2310.09702)

    本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。

    

    随机森林是一种常用的分类和回归方法，在最近几年中提出了许多不同的变体。一个有趣的例子是Mondrian随机森林，其中底层树是根据Mondrian过程构建的。在本文中，我们给出了Mondrian随机森林在回归设置下的估计的中心极限定理。当与偏差表征和一致方差估计器相结合时，这允许进行渐近有效的统计推断，如构建置信区间，对未知的回归函数进行推断。我们还提供了一种去偏过程，用于Mondrian随机森林，使其能够在适当的参数调整下实现$\beta$-H\"older回归函数的最小极大估计速率，对于所有的$\beta$和任意维度。

    Random forests are popular methods for classification and regression, and many different variants have been proposed in recent years. One interesting example is the Mondrian random forest, in which the underlying trees are constructed according to a Mondrian process. In this paper we give a central limit theorem for the estimates made by a Mondrian random forest in the regression setting. When combined with a bias characterization and a consistent variance estimator, this allows one to perform asymptotically valid statistical inference, such as constructing confidence intervals, on the unknown regression function. We also provide a debiasing procedure for Mondrian random forests which allows them to achieve minimax-optimal estimation rates with $\beta$-H\"older regression functions, for all $\beta$ and in arbitrary dimension, assuming appropriate parameter tuning.
    

