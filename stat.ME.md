# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare](https://arxiv.org/abs/2404.02228) | 提出了适用于医疗保健成本效益分析的多元贝叶斯可加回归树扩展，克服了现有模型的局限性，可以处理多个相关结果变量的回归和分类分析 |
| [^2] | [Estimation of Spectral Risk Measure for Left Truncated and Right Censored Data](https://arxiv.org/abs/2402.14322) | 本文提出了基于左截尾和右截尾数据的谱风险度量的估计方法，并建立了相应非参数估计的渐近正态性和Edgeworth展开，对估计量进行了自举法近似分布，并展示具有二阶“准确性”。 |
| [^3] | [Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!.](http://arxiv.org/abs/2206.04902) | 本文介绍了一种半全球框架，用于改进贝叶斯VAR模型的预测性能。该框架替代了传统的全局缩减参数，使用组别特定的缩减参数。通过广泛的模拟研究和实证应用，展示了该框架的优点。在稀疏/密集先验下，预测性能因评估的经济变量和时间框架而异，但动态模型平均法可以缓解这个问题。 |

# 详细

[^1]: 基于贝叶斯可加回归树的医疗保健成本效益分析

    Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare

    [https://arxiv.org/abs/2404.02228](https://arxiv.org/abs/2404.02228)

    提出了适用于医疗保健成本效益分析的多元贝叶斯可加回归树扩展，克服了现有模型的局限性，可以处理多个相关结果变量的回归和分类分析

    

    近年来的理论结果和模拟证据表明，贝叶斯可加回归树是一种非常有效的非参数回归方法。受到在卫生经济学中的成本效益分析的启发，我们提出了适用于具有多个相关结果变量的回归和分类分析的BART的多元扩展。我们的框架通过允许每个个体响应与不同树组相关联，同时处理结果之间的依赖关系，克服了现有多元BART模型的一些主要局限性。在连续结果的情况下，我们的模型本质上是表面无关回归的非参数版本。同样，我们针对二元结果的建议是非参数概括

    arXiv:2404.02228v1 Announce Type: cross  Abstract: In recent years, theoretical results and simulation evidence have shown Bayesian additive regression trees to be a highly-effective method for nonparametric regression. Motivated by cost-effectiveness analyses in health economics, where interest lies in jointly modelling the costs of healthcare treatments and the associated health-related quality of life experienced by a patient, we propose a multivariate extension of BART applicable in regression and classification analyses with several correlated outcome variables. Our framework overcomes some key limitations of existing multivariate BART models by allowing each individual response to be associated with different ensembles of trees, while still handling dependencies between the outcomes. In the case of continuous outcomes, our model is essentially a nonparametric version of seemingly unrelated regression. Likewise, our proposal for binary outcomes is a nonparametric generalisation of
    
[^2]: 对左截尾和右截尾数据的谱风险度量估计

    Estimation of Spectral Risk Measure for Left Truncated and Right Censored Data

    [https://arxiv.org/abs/2402.14322](https://arxiv.org/abs/2402.14322)

    本文提出了基于左截尾和右截尾数据的谱风险度量的估计方法，并建立了相应非参数估计的渐近正态性和Edgeworth展开，对估计量进行了自举法近似分布，并展示具有二阶“准确性”。

    

    左截尾和右截尾数据在保险损失数据中经常遇到，这是由于免赔额和政策限额造成的。风险估计在保险业中是一项重要任务，因为它是确定各种政策条款下的保费的必要步骤。谱风险度量天生具有一致性，并且将风险度量与用户的风险厌恶联系起来。本文研究了基于左截尾和右截尾数据的谱风险度量的估计。我们提出了一个基于产品极限估计的谱风险度量的非参数估计，并为我们提出的估计量建立了渐近正态性。我们还开发了我们提出的估计量的Edgeworth展开。引入自举法来近似我们提出的估计量的分布，并显示其为二阶“准确”。进行蒙特卡洛研究来比较所提出的谱风险度量估计与t

    arXiv:2402.14322v1 Announce Type: cross  Abstract: Left truncated and right censored data are encountered frequently in insurance loss data due to deductibles and policy limits. Risk estimation is an important task in insurance as it is a necessary step for determining premiums under various policy terms. Spectral risk measures are inherently coherent and have the benefit of connecting the risk measure to the user's risk aversion. In this paper we study the estimation of spectral risk measure based on left truncated and right censored data. We propose a non parametric estimator of spectral risk measure using the product limit estimator and establish the asymptotic normality for our proposed estimator. We also develop an Edgeworth expansion of our proposed estimator. The bootstrap is employed to approximate the distribution of our proposed estimator and shown to be second order ``accurate''. Monte Carlo studies are conducted to compare the proposed spectral risk measure estimator with t
    
[^3]: 用贝叶斯VAR模型预测宏观经济数据：稀疏还是密集？要看情况！

    Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!. (arXiv:2206.04902v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.04902](http://arxiv.org/abs/2206.04902)

    本文介绍了一种半全球框架，用于改进贝叶斯VAR模型的预测性能。该框架替代了传统的全局缩减参数，使用组别特定的缩减参数。通过广泛的模拟研究和实证应用，展示了该框架的优点。在稀疏/密集先验下，预测性能因评估的经济变量和时间框架而异，但动态模型平均法可以缓解这个问题。

    

    在建模和预测宏观经济变量时，向量自回归模型（VARs）被广泛应用。然而，在高维情况下，它们容易出现过拟合问题。贝叶斯方法，具体而言是缩减先验方法，已经显示出在提高预测性能方面取得了成功。在本文中，我们引入了半全球框架，其中我们用特定组别的缩减参数替代了传统的全局缩减参数。我们展示了如何将此框架应用于各种缩减先验，如全局-局部先验和随机搜索变量选择先验。我们通过广泛的模拟研究和对美国经济数据进行的实证应用，展示了所提出的框架的优点。此外，我们对正在进行的"稀疏假象"辩论进行了更深入的探讨，发现在稀疏/密集先验下的预测性能在评估的经济变量和时间框架中变化很大。然而，动态模型平均法可以缓解这个问题。

    Vectorautogressions (VARs) are widely applied when it comes to modeling and forecasting macroeconomic variables. In high dimensions, however, they are prone to overfitting. Bayesian methods, more concretely shrinking priors, have shown to be successful in improving prediction performance. In the present paper, we introduce the semi-global framework, in which we replace the traditional global shrinkage parameter with group-specific shrinkage parameters. We show how this framework can be applied to various shrinking priors, such as global-local priors and stochastic search variable selection priors. We demonstrate the virtues of the proposed framework in an extensive simulation study and in an empirical application forecasting data of the US economy. Further, we shed more light on the ongoing ``Illusion of Sparsity'' debate, finding that forecasting performances under sparse/dense priors vary across evaluated economic variables and across time frames. Dynamic model averaging, however, ca
    

