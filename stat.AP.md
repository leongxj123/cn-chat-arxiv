# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Spatiotemporal Prediction with Bayesian Neural Fields](https://arxiv.org/abs/2403.07657) | 该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。 |
| [^2] | [Bayesian Hierarchical Probabilistic Forecasting of Intraday Electricity Prices](https://arxiv.org/abs/2403.05441) | 该研究首次提出了为德国连续日内市场交易的电力价格进行贝叶斯预测，考虑了参数不确定性，并在2022年的电力价格验证中取得了统计显著的改进。 |

# 详细

[^1]: 使用贝叶斯神经场进行可扩展的时空预测

    Scalable Spatiotemporal Prediction with Bayesian Neural Fields

    [https://arxiv.org/abs/2403.07657](https://arxiv.org/abs/2403.07657)

    该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。

    

    时空数据集由空间参考的时间序列表示，广泛应用于许多科学和商业智能领域，例如空气污染监测，疾病跟踪和云需求预测。随着现代数据集规模和复杂性的不断增加，需要新的统计方法来捕捉复杂的时空动态并处理大规模预测问题。本研究介绍了Bayesian Neural Field (BayesNF)，这是一个用于推断时空域上丰富概率分布的通用领域统计模型，可用于包括预测、插值和变异分析在内的数据分析任务。BayesNF将用于高容量函数估计的新型深度神经网络架构与用于鲁棒不确定性量化的分层贝叶斯推断相结合。通过在定义先验分布方面进行序列化

    arXiv:2403.07657v1 Announce Type: cross  Abstract: Spatiotemporal datasets, which consist of spatially-referenced time series, are ubiquitous in many scientific and business-intelligence applications, such as air pollution monitoring, disease tracking, and cloud-demand forecasting. As modern datasets continue to increase in size and complexity, there is a growing need for new statistical methods that are flexible enough to capture complex spatiotemporal dynamics and scalable enough to handle large prediction problems. This work presents the Bayesian Neural Field (BayesNF), a domain-general statistical model for inferring rich probability distributions over a spatiotemporal domain, which can be used for data-analysis tasks including forecasting, interpolation, and variography. BayesNF integrates a novel deep neural network architecture for high-capacity function estimation with hierarchical Bayesian inference for robust uncertainty quantification. By defining the prior through a sequenc
    
[^2]: 基于贝叶斯层次概率的日内电力价格预测

    Bayesian Hierarchical Probabilistic Forecasting of Intraday Electricity Prices

    [https://arxiv.org/abs/2403.05441](https://arxiv.org/abs/2403.05441)

    该研究首次提出了为德国连续日内市场交易的电力价格进行贝叶斯预测，考虑了参数不确定性，并在2022年的电力价格验证中取得了统计显著的改进。

    

    我们首次提出了对德国连续日内市场交易的电力价格进行贝叶斯预测的研究，充分考虑参数不确定性。我们的目标变量是IDFull价格指数，预测以后验预测分布的形式给出。我们使用了2022年极度波动的电力价格进行验证，在之前几乎没有成为预测研究对象。作为基准模型，我们使用了预测创建时的所有可用日内交易来计算IDFull的当前值。根据弱式有效假设，从最后价格信息建立的基准无法显著改善。然而，我们观察到在点度量和概率评分方面存在着统计显著的改进。最后，我们挑战了在电力价格预测中使用LASSO进行特征选择的宣布的黄金标准。

    arXiv:2403.05441v1 Announce Type: cross  Abstract: We present a first study of Bayesian forecasting of electricity prices traded on the German continuous intraday market which fully incorporates parameter uncertainty. Our target variable is the IDFull price index, forecasts are given in terms of posterior predictive distributions. For validation we use the exceedingly volatile electricity prices of 2022, which have hardly been the subject of forecasting studies before. As a benchmark model, we use all available intraday transactions at the time of forecast creation to compute a current value for the IDFull. According to the weak-form efficiency hypothesis, it would not be possible to significantly improve this benchmark built from last price information. We do, however, observe statistically significant improvement in terms of both point measures and probability scores. Finally, we challenge the declared gold standard of using LASSO for feature selection in electricity price forecastin
    

