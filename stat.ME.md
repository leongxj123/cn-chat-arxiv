# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms](https://arxiv.org/abs/2402.04952) | 本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。 |
| [^2] | [Latent Gaussian dynamic factor modeling and forecasting for multivariate count time series.](http://arxiv.org/abs/2307.10454) | 本文提出了一种针对多元计数时间序列的潜在高斯动态因子建模与预测方法，通过计数和底层高斯模型的二阶特性进行估计，并利用粒子顺序蒙特卡洛方法进行预测。 |

# 详细

[^1]: 评估因果推断算法的马尔科夫等价类指标

    Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms

    [https://arxiv.org/abs/2402.04952](https://arxiv.org/abs/2402.04952)

    本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。

    

    许多最先进的因果推断方法旨在生成一个输出图，该图编码了生成数据过程的因果图的图形分离和连接陈述。在本文中，我们认为，对合成数据的因果推断方法进行评估应该包括分析该方法的输出与真实情况的分离/连接程度，以衡量这一明确目标的实现情况。我们证明现有的评估指标不能准确捕捉到两个因果图的分离/连接差异，并引入了三个新的距离度量指标，即s/c距离、马尔科夫距离和忠实度距离，以解决这个问题。我们通过玩具示例、实证实验和伪代码来补充我们的理论分析。

    Many state-of-the-art causal discovery methods aim to generate an output graph that encodes the graphical separation and connection statements of the causal graph that underlies the data-generating process. In this work, we argue that an evaluation of a causal discovery method against synthetic data should include an analysis of how well this explicit goal is achieved by measuring how closely the separations/connections of the method's output align with those of the ground truth. We show that established evaluation measures do not accurately capture the difference in separations/connections of two causal graphs, and we introduce three new measures of distance called s/c-distance, Markov distance and Faithfulness distance that address this shortcoming. We complement our theoretical analysis with toy examples, empirical experiments and pseudocode.
    
[^2]: 针对多元计数时间序列的潜在高斯动态因子建模与预测

    Latent Gaussian dynamic factor modeling and forecasting for multivariate count time series. (arXiv:2307.10454v1 [stat.ME])

    [http://arxiv.org/abs/2307.10454](http://arxiv.org/abs/2307.10454)

    本文提出了一种针对多元计数时间序列的潜在高斯动态因子建模与预测方法，通过计数和底层高斯模型的二阶特性进行估计，并利用粒子顺序蒙特卡洛方法进行预测。

    

    本文考虑了一种基于高斯动态因子模型的多元计数时间序列模型的估计和预测方法，该方法基于计数和底层高斯模型的二阶特性，并适用于模型维数大于样本长度的情况。此外，本文提出了用于模型选择的新型交叉验证方案。预测通过基于粒子的顺序蒙特卡洛方法进行，利用卡尔曼滤波技术。还进行了模拟研究和应用分析。

    This work considers estimation and forecasting in a multivariate count time series model based on a copula-type transformation of a Gaussian dynamic factor model. The estimation is based on second-order properties of the count and underlying Gaussian models and applies to the case where the model dimension is larger than the sample length. In addition, novel cross-validation schemes are suggested for model selection. The forecasting is carried out through a particle-based sequential Monte Carlo, leveraging Kalman filtering techniques. A simulation study and an application are also considered.
    

