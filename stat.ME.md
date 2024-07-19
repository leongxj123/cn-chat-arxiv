# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Spatiotemporal Prediction with Bayesian Neural Fields](https://arxiv.org/abs/2403.07657) | 该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。 |
| [^2] | [Latent Gaussian dynamic factor modeling and forecasting for multivariate count time series.](http://arxiv.org/abs/2307.10454) | 本文提出了一种针对多元计数时间序列的潜在高斯动态因子建模与预测方法，通过计数和底层高斯模型的二阶特性进行估计，并利用粒子顺序蒙特卡洛方法进行预测。 |

# 详细

[^1]: 使用贝叶斯神经场进行可扩展的时空预测

    Scalable Spatiotemporal Prediction with Bayesian Neural Fields

    [https://arxiv.org/abs/2403.07657](https://arxiv.org/abs/2403.07657)

    该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。

    

    时空数据集由空间参考的时间序列表示，广泛应用于许多科学和商业智能领域，例如空气污染监测，疾病跟踪和云需求预测。随着现代数据集规模和复杂性的不断增加，需要新的统计方法来捕捉复杂的时空动态并处理大规模预测问题。本研究介绍了Bayesian Neural Field (BayesNF)，这是一个用于推断时空域上丰富概率分布的通用领域统计模型，可用于包括预测、插值和变异分析在内的数据分析任务。BayesNF将用于高容量函数估计的新型深度神经网络架构与用于鲁棒不确定性量化的分层贝叶斯推断相结合。通过在定义先验分布方面进行序列化

    arXiv:2403.07657v1 Announce Type: cross  Abstract: Spatiotemporal datasets, which consist of spatially-referenced time series, are ubiquitous in many scientific and business-intelligence applications, such as air pollution monitoring, disease tracking, and cloud-demand forecasting. As modern datasets continue to increase in size and complexity, there is a growing need for new statistical methods that are flexible enough to capture complex spatiotemporal dynamics and scalable enough to handle large prediction problems. This work presents the Bayesian Neural Field (BayesNF), a domain-general statistical model for inferring rich probability distributions over a spatiotemporal domain, which can be used for data-analysis tasks including forecasting, interpolation, and variography. BayesNF integrates a novel deep neural network architecture for high-capacity function estimation with hierarchical Bayesian inference for robust uncertainty quantification. By defining the prior through a sequenc
    
[^2]: 针对多元计数时间序列的潜在高斯动态因子建模与预测

    Latent Gaussian dynamic factor modeling and forecasting for multivariate count time series. (arXiv:2307.10454v1 [stat.ME])

    [http://arxiv.org/abs/2307.10454](http://arxiv.org/abs/2307.10454)

    本文提出了一种针对多元计数时间序列的潜在高斯动态因子建模与预测方法，通过计数和底层高斯模型的二阶特性进行估计，并利用粒子顺序蒙特卡洛方法进行预测。

    

    本文考虑了一种基于高斯动态因子模型的多元计数时间序列模型的估计和预测方法，该方法基于计数和底层高斯模型的二阶特性，并适用于模型维数大于样本长度的情况。此外，本文提出了用于模型选择的新型交叉验证方案。预测通过基于粒子的顺序蒙特卡洛方法进行，利用卡尔曼滤波技术。还进行了模拟研究和应用分析。

    This work considers estimation and forecasting in a multivariate count time series model based on a copula-type transformation of a Gaussian dynamic factor model. The estimation is based on second-order properties of the count and underlying Gaussian models and applies to the case where the model dimension is larger than the sample length. In addition, novel cross-validation schemes are suggested for model selection. The forecasting is carried out through a particle-based sequential Monte Carlo, leveraging Kalman filtering techniques. A simulation study and an application are also considered.
    

