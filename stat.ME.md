# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictive Inference in Multi-environment Scenarios](https://arxiv.org/abs/2403.16336) | 本研究提出了在多环境预测问题中构建有效置信区间和置信集的方法，并展示了一种新的调整方法以适应问题难度，从而减少预测集大小，这在神经感应和物种分类数据集中的实际表现中得到验证。 |
| [^2] | [Neural Networks for Extreme Quantile Regression with an Application to Forecasting of Flood Risk.](http://arxiv.org/abs/2208.07590) | 本文提出了一种结合神经网络和极值理论的EQRN模型，它能够在存在复杂预测变量相关性的情况下进行外推，并且能够应用于洪水风险预测中，提供一天前回归水平和超出概率的预测。 |
| [^3] | [Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!.](http://arxiv.org/abs/2206.04902) | 本文介绍了一种半全球框架，用于改进贝叶斯VAR模型的预测性能。该框架替代了传统的全局缩减参数，使用组别特定的缩减参数。通过广泛的模拟研究和实证应用，展示了该框架的优点。在稀疏/密集先验下，预测性能因评估的经济变量和时间框架而异，但动态模型平均法可以缓解这个问题。 |

# 详细

[^1]: 多环境场景中的预测推断

    Predictive Inference in Multi-environment Scenarios

    [https://arxiv.org/abs/2403.16336](https://arxiv.org/abs/2403.16336)

    本研究提出了在多环境预测问题中构建有效置信区间和置信集的方法，并展示了一种新的调整方法以适应问题难度，从而减少预测集大小，这在神经感应和物种分类数据集中的实际表现中得到验证。

    

    我们解决了在跨多个环境的预测问题中构建有效置信区间和置信集的挑战。我们研究了适用于这些问题的两种覆盖类型，扩展了Jackknife和分裂一致方法，展示了如何在这种非传统的层次数据生成场景中获得无分布覆盖。我们的贡献还包括对非实值响应设置的扩展，以及这些一般问题中预测推断的一致性理论。我们展示了一种新的调整方法，以适应问题难度，这适用于具有层次数据的预测推断的现有方法以及我们开发的方法；这通过神经化学感应和物种分类数据集评估了这些方法的实际性能。

    arXiv:2403.16336v1 Announce Type: cross  Abstract: We address the challenge of constructing valid confidence intervals and sets in problems of prediction across multiple environments. We investigate two types of coverage suitable for these problems, extending the jackknife and split-conformal methods to show how to obtain distribution-free coverage in such non-traditional, hierarchical data-generating scenarios. Our contributions also include extensions for settings with non-real-valued responses and a theory of consistency for predictive inference in these general problems. We demonstrate a novel resizing method to adapt to problem difficulty, which applies both to existing approaches for predictive inference with hierarchical data and the methods we develop; this reduces prediction set sizes using limited information from the test environment, a key to the methods' practical performance, which we evaluate through neurochemical sensing and species classification datasets.
    
[^2]: 极端分位数回归的神经网络与洪水风险预测应用

    Neural Networks for Extreme Quantile Regression with an Application to Forecasting of Flood Risk. (arXiv:2208.07590v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.07590](http://arxiv.org/abs/2208.07590)

    本文提出了一种结合神经网络和极值理论的EQRN模型，它能够在存在复杂预测变量相关性的情况下进行外推，并且能够应用于洪水风险预测中，提供一天前回归水平和超出概率的预测。

    

    针对极端事件的风险评估需要准确估计超出历史观测范围的高分位数。当风险依赖于观测预测变量的值时，回归技术用于在预测空间中进行插值。我们提出了EQRN模型，它将神经网络和极值理论的工具结合起来，形成一种能够在复杂预测变量相关性存在的情况下进行外推的方法。神经网络可以自然地将数据中的附加结构纳入其中。我们开发了EQRN的循环版本，能够捕捉时间序列中复杂的顺序相关性。我们将这种方法应用于瑞士Aare流域的洪水风险预测。它利用空间和时间上的多个协变量信息，提供一天前回归水平和超出概率的预测。这个输出补充了传统极值分析的静态回归水平，并且预测能够适应分布变化。

    Risk assessment for extreme events requires accurate estimation of high quantiles that go beyond the range of historical observations. When the risk depends on the values of observed predictors, regression techniques are used to interpolate in the predictor space. We propose the EQRN model that combines tools from neural networks and extreme value theory into a method capable of extrapolation in the presence of complex predictor dependence. Neural networks can naturally incorporate additional structure in the data. We develop a recurrent version of EQRN that is able to capture complex sequential dependence in time series. We apply this method to forecasting of flood risk in the Swiss Aare catchment. It exploits information from multiple covariates in space and time to provide one-day-ahead predictions of return levels and exceedances probabilities. This output complements the static return level from a traditional extreme value analysis and the predictions are able to adapt to distribu
    
[^3]: 用贝叶斯VAR模型预测宏观经济数据：稀疏还是密集？要看情况！

    Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!. (arXiv:2206.04902v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.04902](http://arxiv.org/abs/2206.04902)

    本文介绍了一种半全球框架，用于改进贝叶斯VAR模型的预测性能。该框架替代了传统的全局缩减参数，使用组别特定的缩减参数。通过广泛的模拟研究和实证应用，展示了该框架的优点。在稀疏/密集先验下，预测性能因评估的经济变量和时间框架而异，但动态模型平均法可以缓解这个问题。

    

    在建模和预测宏观经济变量时，向量自回归模型（VARs）被广泛应用。然而，在高维情况下，它们容易出现过拟合问题。贝叶斯方法，具体而言是缩减先验方法，已经显示出在提高预测性能方面取得了成功。在本文中，我们引入了半全球框架，其中我们用特定组别的缩减参数替代了传统的全局缩减参数。我们展示了如何将此框架应用于各种缩减先验，如全局-局部先验和随机搜索变量选择先验。我们通过广泛的模拟研究和对美国经济数据进行的实证应用，展示了所提出的框架的优点。此外，我们对正在进行的"稀疏假象"辩论进行了更深入的探讨，发现在稀疏/密集先验下的预测性能在评估的经济变量和时间框架中变化很大。然而，动态模型平均法可以缓解这个问题。

    Vectorautogressions (VARs) are widely applied when it comes to modeling and forecasting macroeconomic variables. In high dimensions, however, they are prone to overfitting. Bayesian methods, more concretely shrinking priors, have shown to be successful in improving prediction performance. In the present paper, we introduce the semi-global framework, in which we replace the traditional global shrinkage parameter with group-specific shrinkage parameters. We show how this framework can be applied to various shrinking priors, such as global-local priors and stochastic search variable selection priors. We demonstrate the virtues of the proposed framework in an extensive simulation study and in an empirical application forecasting data of the US economy. Further, we shed more light on the ongoing ``Illusion of Sparsity'' debate, finding that forecasting performances under sparse/dense priors vary across evaluated economic variables and across time frames. Dynamic model averaging, however, ca
    

