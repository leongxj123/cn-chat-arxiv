# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Assessing Heterogeneity of Treatment Effects.](http://arxiv.org/abs/2306.15048) | 该论文介绍了一种评估治疗效果异质性的方法，通过使用治疗组和对照组结果的分位数范围，即使平均效果不显著，也可以提供有用的信息。 |
| [^2] | [Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information.](http://arxiv.org/abs/2210.00116) | 本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。 |

# 详细

[^1]: 评估治疗效果的异质性

    Assessing Heterogeneity of Treatment Effects. (arXiv:2306.15048v1 [econ.EM])

    [http://arxiv.org/abs/2306.15048](http://arxiv.org/abs/2306.15048)

    该论文介绍了一种评估治疗效果异质性的方法，通过使用治疗组和对照组结果的分位数范围，即使平均效果不显著，也可以提供有用的信息。

    

    异质性治疗效果在经济学中非常重要，但是其评估常常受到个体治疗效果无法确定的困扰。例如，我们可能希望评估保险对本来不健康的人的健康影响，但是只给不健康的人买保险是不可行的，因此这些人的因果效应无法确定。又或者，我们可能对最低工资上涨中赢家的份额感兴趣，但是在没有观察到反事实的情况下，赢家也无法确定。这种异质性常常通过分位数治疗效果来评估，但其解释并不清晰，结论有时也不一致。我们展示了通过治疗组和对照组结果的分位数，这些数值范围是可以确定的，即使平均治疗效果并不显著，它们仍然可以提供有用信息。两个应用实例展示了这些范围如何帮助我们了解治疗效果的异质性。

    Treatment effect heterogeneity is of major interest in economics, but its assessment is often hindered by the fundamental lack of identification of the individual treatment effects. For example, we may want to assess the effect of insurance on the health of otherwise unhealthy individuals, but it is infeasible to insure only the unhealthy, and thus the causal effects for those are not identified. Or, we may be interested in the shares of winners from a minimum wage increase, while without observing the counterfactual, the winners are not identified. Such heterogeneity is often assessed by quantile treatment effects, which do not come with clear interpretation and the takeaway can sometimes be equivocal. We show that, with the quantiles of the treated and control outcomes, the ranges of these quantities are identified and can be informative even when the average treatment effects are not significant. Two applications illustrate how these ranges can inform us about heterogeneity of the t
    
[^2]: 利用变分因果推断和精细关系信息预测细胞响应

    Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information. (arXiv:2210.00116v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.00116](http://arxiv.org/abs/2210.00116)

    本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。

    

    预测细胞在干扰下的响应可能为药物研发和个性化治疗带来重要好处。在本研究中，我们提出了一种新的图形变分贝叶斯因果推断框架，预测细胞在反事实干扰下（即细胞未真实接收的干扰）的基因表达，利用代表生物学知识的基因调控网络（GRN）信息来辅助个性化细胞响应预测。我们还针对数据自适应GRN开发了邻接矩阵更新技术用于图卷积网络的预训练，在模型性能上提供了更多的基因关系洞见。

    Predicting the responses of a cell under perturbations may bring important benefits to drug discovery and personalized therapeutics. In this work, we propose a novel graph variational Bayesian causal inference framework to predict a cell's gene expressions under counterfactual perturbations (perturbations that this cell did not factually receive), leveraging information representing biological knowledge in the form of gene regulatory networks (GRNs) to aid individualized cellular response predictions. Aiming at a data-adaptive GRN, we also developed an adjacency matrix updating technique for graph convolutional networks and used it to refine GRNs during pre-training, which generated more insights on gene relations and enhanced model performance. Additionally, we propose a robust estimator within our framework for the asymptotically efficient estimation of marginal perturbation effect, which is yet to be carried out in previous works. With extensive experiments, we exhibited the advanta
    

