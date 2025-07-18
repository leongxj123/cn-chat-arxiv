# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrated path stability selection](https://arxiv.org/abs/2403.15877) | 该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。 |

# 详细

[^1]: 集成路径稳定选择

    Integrated path stability selection

    [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)

    该论文提出了一种基于集成稳定路径的新稳定选择方法，能够在实践中提高特征选择的灵敏度并更好地校准目标假阳性数量。

    

    稳定选择是一种广泛用于改善特征选择算法性能的方法。然而，已发现稳定选择过于保守，导致灵敏度较低。此外，对期望的假阳性数量的理论界限E(FP)相对较松，难以知道实践中会有多少假阳性。在本文中，我们提出一种基于集成稳定路径而非最大化稳定路径的新方法。这产生了对E(FP)更紧密的界限，导致实践中具有更高灵敏度的特征选择标准，并且在与目标E(FP)匹配方面更好地校准。我们提出的方法与原始稳定选择算法需要相同数量的计算，且仅需要用户指定一个输入参数，即E(FP)的目标值。我们提供了性能的理论界限。

    arXiv:2403.15877v1 Announce Type: cross  Abstract: Stability selection is a widely used method for improving the performance of feature selection algorithms. However, stability selection has been found to be highly conservative, resulting in low sensitivity. Further, the theoretical bound on the expected number of false positives, E(FP), is relatively loose, making it difficult to know how many false positives to expect in practice. In this paper, we introduce a novel method for stability selection based on integrating the stability paths rather than maximizing over them. This yields a tighter bound on E(FP), resulting in a feature selection criterion that has higher sensitivity in practice and is better calibrated in terms of matching the target E(FP). Our proposed method requires the same amount of computation as the original stability selection algorithm, and only requires the user to specify one input parameter, a target value for E(FP). We provide theoretical bounds on performance
    

