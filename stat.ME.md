# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heteroscedastic sparse high-dimensional linear regression with a partitioned empirical Bayes ECM algorithm.](http://arxiv.org/abs/2309.08783) | 本文提出了一种解决高维度稀疏线性回归中异方差问题的方法，通过基于分区经验贝叶斯ECM算法的异方差高维度线性回归模型来实现。这个模型可以处理残差方差不恒定的情况，并且可以使用插值的经验贝叶斯估计超参数来灵活地调整方差模型。 |
| [^2] | [UTOPIA: Universally Trainable Optimal Prediction Intervals Aggregation.](http://arxiv.org/abs/2306.16549) | 本论文提出了一种新的技术，称为UTOPIA，用于聚合多个预测区间以减小其宽度并保证覆盖率。该方法基于线性或凸规划，易于实现，适用范围广泛。 |
| [^3] | [Counterfactual Generative Models for Time-Varying Treatments.](http://arxiv.org/abs/2305.15742) | 本文研究了时间变量处理情况下的反事实生成模型，能够捕捉整个反事实分布，并且能够有效推断反事实分布的某些统计量，适用于医疗保健和公共政策制定领域。 |
| [^4] | [Factorized Fusion Shrinkage for Dynamic Relational Data.](http://arxiv.org/abs/2210.00091) | 本文提出了一种动态关系数据的分解融合压缩模型，通过对分解矩阵的行向量的逐次差值施加全局-局部压缩先验获得收缩，并具有许多有利的性质。 |

# 详细

[^1]: 高维度稀疏线性回归中的异方差问题及基于分区经验贝叶斯ECM算法的解决方法

    Heteroscedastic sparse high-dimensional linear regression with a partitioned empirical Bayes ECM algorithm. (arXiv:2309.08783v1 [stat.ME])

    [http://arxiv.org/abs/2309.08783](http://arxiv.org/abs/2309.08783)

    本文提出了一种解决高维度稀疏线性回归中异方差问题的方法，通过基于分区经验贝叶斯ECM算法的异方差高维度线性回归模型来实现。这个模型可以处理残差方差不恒定的情况，并且可以使用插值的经验贝叶斯估计超参数来灵活地调整方差模型。

    

    高维度数据的稀疏线性回归方法通常假设残差具有常数方差。当这一假设被违背时，会导致估计系数的偏差，预测区间长度不合适以及增加I型错误。本文提出一种基于分区经验贝叶斯期望条件最大化(H-PROBE)算法的异方差高维度线性回归模型。H-PROBE是一种计算效率高的最大后验估计方法，基于参数扩展的期望条件最大化(PX-ECM)算法。它通过插值的经验贝叶斯估计超参数，在回归参数上假设最小。方差模型使用了多元对数伽马分布理论的最新进展，并可以包含假设会影响异质性的协变量。我们的方法的动机是通过T2高分辨率神经影像研究与失语指数(AQ)相关性。

    Sparse linear regression methods for high-dimensional data often assume that residuals have constant variance. When this assumption is violated, it can lead to bias in estimated coefficients, prediction intervals with improper length, and increased type I errors. This paper proposes a heteroscedastic (H) high-dimensional linear regression model through a partitioned empirical Bayes Expectation Conditional Maximization (H-PROBE) algorithm. H-PROBE is a computationally efficient maximum a posteriori (MAP) estimation approach based on a Parameter-Expanded Expectation-Conditional-Maximization (PX-ECM) algorithm. It requires minimal prior assumptions on the regression parameters through plug-in empirical Bayes estimates of hyperparameters. The variance model uses recent advances in multivariate log-Gamma distribution theory and can include covariates hypothesized to impact heterogeneity. The motivation of our approach is a study relating Aphasia Quotient (AQ) to high-resolution T2 neuroimag
    
[^2]: UTOPIA：通用可训练的最优预测区间聚合

    UTOPIA: Universally Trainable Optimal Prediction Intervals Aggregation. (arXiv:2306.16549v1 [stat.ME])

    [http://arxiv.org/abs/2306.16549](http://arxiv.org/abs/2306.16549)

    本论文提出了一种新的技术，称为UTOPIA，用于聚合多个预测区间以减小其宽度并保证覆盖率。该方法基于线性或凸规划，易于实现，适用范围广泛。

    

    预测的不确定性量化是一个有趣的问题，在生物医学科学、经济研究和天气预报等各个领域有重要的应用。构建预测区间的方法有很多，如分位数回归和一致性预测等。然而，模型错误规定（尤其是在高维情况下）或次优的构造通常会导致有偏或过宽的预测区间。在本文中，我们提出了一种新颖且广泛适用的技术，用于聚合多个预测区间以最小化预测带的平均宽度和覆盖保证，称为通用可训练的最优预测区间聚合（UTOPIA）。该方法还允许我们根据基本的基函数直接构建预测带。我们的方法基于线性或凸规划，易于实现。我们提出的所有方法都得到了实验证明。

    Uncertainty quantification for prediction is an intriguing problem with significant applications in various fields, such as biomedical science, economic studies, and weather forecasts. Numerous methods are available for constructing prediction intervals, such as quantile regression and conformal predictions, among others. Nevertheless, model misspecification (especially in high-dimension) or sub-optimal constructions can frequently result in biased or unnecessarily-wide prediction intervals. In this paper, we propose a novel and widely applicable technique for aggregating multiple prediction intervals to minimize the average width of the prediction band along with coverage guarantee, called Universally Trainable Optimal Predictive Intervals Aggregation (UTOPIA). The method also allows us to directly construct predictive bands based on elementary basis functions. Our approach is based on linear or convex programming which is easy to implement. All of our proposed methodologies are suppo
    
[^3]: 时间变化处理的反事实生成模型

    Counterfactual Generative Models for Time-Varying Treatments. (arXiv:2305.15742v1 [stat.ML])

    [http://arxiv.org/abs/2305.15742](http://arxiv.org/abs/2305.15742)

    本文研究了时间变量处理情况下的反事实生成模型，能够捕捉整个反事实分布，并且能够有效推断反事实分布的某些统计量，适用于医疗保健和公共政策制定领域。

    

    估计平均因果效应是测试新疗法的常用做法。然而，平均效应会掩盖反事实分布中重要的个体特征，可能会引起安全、公平和道德方面的担忧。这个问题在时间设置中更加严重，因为处理是时序的和时变的，对反事实分布产生了错综复杂的影响。本文提出了一种新的条件生成建模方法，以捕获整个反事实分布，允许对反事实分布的某些统计量进行有效推断。这使得所提出的方法尤其适用于医疗保健和公共政策制定领域。我们的生成建模方法通过边际结构模型谨慎地解决了观察数据和目标反事实分布之间的分布不匹配。在合成和真实数据上，我们的方法优于现有的基线方法。

    Estimating average causal effects is a common practice to test new treatments. However, the average effect ''masks'' important individual characteristics in the counterfactual distribution, which may lead to safety, fairness, and ethical concerns. This issue is exacerbated in the temporal setting, where the treatment is sequential and time-varying, leading to an intricate influence on the counterfactual distribution. In this paper, we propose a novel conditional generative modeling approach to capture the whole counterfactual distribution, allowing efficient inference on certain statistics of the counterfactual distribution. This makes the proposed approach particularly suitable for healthcare and public policy making. Our generative modeling approach carefully tackles the distribution mismatch in the observed data and the targeted counterfactual distribution via a marginal structural model. Our method outperforms state-of-the-art baselines on both synthetic and real data.
    
[^4]: 动态关系数据的分解融合压缩模型

    Factorized Fusion Shrinkage for Dynamic Relational Data. (arXiv:2210.00091v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2210.00091](http://arxiv.org/abs/2210.00091)

    本文提出了一种动态关系数据的分解融合压缩模型，通过对分解矩阵的行向量的逐次差值施加全局-局部压缩先验获得收缩，并具有许多有利的性质。

    

    现代数据科学应用经常涉及具有动态结构的复杂关系数据。此类动态关系数据的突变通常出现在由于干预而经历制度变化的系统中。在这种情况下，我们考虑一种分解融合压缩模型，其中所有分解因子都被动态地收缩到组内融合结构，收缩通过对分解矩阵的行向量的逐次差值施加全局-局部压缩先验来获得。所提出的先验在估计的动态潜在因子的比较和聚类方面具有许多有利的性质。比较估计的潜在因子涉及相邻和长期比较，考虑到比较的时间范围作为变量。在某些条件下，我们证明后验分布达到最小化最大风险，直到对数因子。在计算方面，我们提出了一个结构化的均值场变分方法。

    Modern data science applications often involve complex relational data with dynamic structures. An abrupt change in such dynamic relational data is typically observed in systems that undergo regime changes due to interventions. In such a case, we consider a factorized fusion shrinkage model in which all decomposed factors are dynamically shrunk towards group-wise fusion structures, where the shrinkage is obtained by applying global-local shrinkage priors to the successive differences of the row vectors of the factorized matrices. The proposed priors enjoy many favorable properties in comparison and clustering of the estimated dynamic latent factors. Comparing estimated latent factors involves both adjacent and long-term comparisons, with the time range of comparison considered as a variable. Under certain conditions, we demonstrate that the posterior distribution attains the minimax optimal rate up to logarithmic factors. In terms of computation, we present a structured mean-field vari
    

