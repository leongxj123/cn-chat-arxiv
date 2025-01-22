# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic CoVaR Modeling](https://arxiv.org/abs/2206.14275) | 提出了用于风险价值（VaR）和CoVaR的联合动态预测模型，引入了一种新的参数估计方法，并在美国大型银行的实证分析中展示了其优越性。 |
| [^2] | [Explainable Performance: Measuring the Driving Forces of Predictive Performance.](http://arxiv.org/abs/2212.05866) | XPER方法能衡量输入特征对模型预测性能的具体贡献，并可用于处理异质性问题，构建同质化个体群体，从而提高预测精度。 |

# 详细

[^1]: 动态CoVaR建模

    Dynamic CoVaR Modeling

    [https://arxiv.org/abs/2206.14275](https://arxiv.org/abs/2206.14275)

    提出了用于风险价值（VaR）和CoVaR的联合动态预测模型，引入了一种新的参数估计方法，并在美国大型银行的实证分析中展示了其优越性。

    

    CoVaR（条件风险价值）是一种流行的系统风险度量方法，在经济学和金融领域被广泛使用。本文提出了用于风险价值（VaR）和CoVaR的联合动态预测模型。我们还介绍了一种基于最近提出的VaR和CoVaR对的双变量评分函数的模型参数的两步M估计量。我们证明了参数估计量的一致性和渐近正态性，并分析了它在模拟中的有限样本性质。最后，我们将我们的动态预测模型的一个特定子类应用于美国大型银行的对数收益率。结果表明，我们的CoCAViaR模型产生的CoVaR预测优于当前基准模型发布的预测。

    arXiv:2206.14275v3 Announce Type: replace  Abstract: The popular systemic risk measure CoVaR (conditional Value-at-Risk) is widely used in economics and finance. Formally, it is defined as a large quantile of one variable (e.g., losses in the financial system) conditional on some other variable (e.g., losses in a bank's shares) being in distress. In this article, we propose joint dynamic forecasting models for the Value-at-Risk (VaR) and CoVaR. We also introduce a two-step M-estimator for the model parameters drawing on recently proposed bivariate scoring functions for the pair (VaR, CoVaR). We prove consistency and asymptotic normality of our parameter estimator and analyze its finite-sample properties in simulations. Finally, we apply a specific subclass of our dynamic forecasting models, which we call CoCAViaR models, to log-returns of large US banks. It is shown that our CoCAViaR models generate CoVaR predictions that are superior to forecasts issued from current benchmark models.
    
[^2]: 可解释的性能：衡量预测性能的驱动力

    Explainable Performance: Measuring the Driving Forces of Predictive Performance. (arXiv:2212.05866v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.05866](http://arxiv.org/abs/2212.05866)

    XPER方法能衡量输入特征对模型预测性能的具体贡献，并可用于处理异质性问题，构建同质化个体群体，从而提高预测精度。

    

    我们引入了XPER（eXplainable PERformance）方法来衡量输入特征对模型预测性能的具体贡献。我们的方法在理论上基于Shapley值，既不依赖于模型，也不依赖于性能度量。此外，XPER可在模型级别或个体级别实现。我们证明XPER具有标准解释性方法（SHAP）的特殊情况。在贷款违约预测应用中，我们展示了如何利用XPER处理异质性问题，并显著提高样本外性能。为此，我们通过基于个体XPER值对他们进行聚类来构建同质化的个体群体。我们发现估计群体特定的模型比一个模型适用于所有个体具有更高的预测精度。

    We introduce the XPER (eXplainable PERformance) methodology to measure the specific contribution of the input features to the predictive performance of a model. Our methodology is theoretically grounded on Shapley values and is both model-agnostic and performance metric-agnostic. Furthermore, XPER can be implemented either at the model level or at the individual level. We demonstrate that XPER has as a special case the standard explainability method in machine learning (SHAP). In a loan default forecasting application, we show how XPER can be used to deal with heterogeneity issues and significantly boost out-of-sample performance. To do so, we build homogeneous groups of individuals by clustering them based on their individual XPER values. We find that estimating group-specific models yields a much higher predictive accuracy than with a one-fits-all model.
    

