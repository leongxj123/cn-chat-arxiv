# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.](http://arxiv.org/abs/2401.15299) | SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。 |
| [^2] | [Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective.](http://arxiv.org/abs/2311.02043) | 本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。 |

# 详细

[^1]: SupplyGraph: 使用图神经网络进行供应链规划的基准数据集

    SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks. (arXiv:2401.15299v1 [cs.LG])

    [http://arxiv.org/abs/2401.15299](http://arxiv.org/abs/2401.15299)

    SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。

    

    图神经网络（GNNs）在不同领域如运输、生物信息学、语言处理和计算机视觉中取得了重要进展。然而，在将GNNs应用于供应链网络方面，目前尚缺乏研究。供应链网络在结构上类似于图形，使其成为应用GNN方法的理想选择。这为优化、预测和解决供应链问题开辟了无限可能。然而，此方法的一个主要障碍在于缺乏真实世界的基准数据集以促进使用GNN来研究和解决供应链问题。为了解决这个问题，我们提供了一个来自孟加拉国一家领先的快速消费品公司的实际基准数据集，该数据集侧重于用于生产目的的供应链规划的时间任务。该数据集包括时间数据作为节点特征，以实现销售预测、生产计划和故障识别。

    Graph Neural Networks (GNNs) have gained traction across different domains such as transportation, bio-informatics, language processing, and computer vision. However, there is a noticeable absence of research on applying GNNs to supply chain networks. Supply chain networks are inherently graph-like in structure, making them prime candidates for applying GNN methodologies. This opens up a world of possibilities for optimizing, predicting, and solving even the most complex supply chain problems. A major setback in this approach lies in the absence of real-world benchmark datasets to facilitate the research and resolution of supply chain problems using GNNs. To address the issue, we present a real-world benchmark dataset for temporal tasks, obtained from one of the leading FMCG companies in Bangladesh, focusing on supply chain planning for production purposes. The dataset includes temporal data as node features to enable sales predictions, production planning, and the identification of fa
    
[^2]: 基于子集选择的贝叶斯分位回归：后验总结视角

    Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective. (arXiv:2311.02043v1 [stat.ME])

    [http://arxiv.org/abs/2311.02043](http://arxiv.org/abs/2311.02043)

    本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。

    

    分位回归是一种强大的工具，用于推断协变量如何影响响应分布的特定分位数。现有方法要么分别估计每个感兴趣分位数的条件分位数，要么使用半参数或非参数模型估计整个条件分布。前者经常产生不适合实际数据的模型，并且不在分位数之间共享信息，而后者则以复杂且受限制的模型为特点，难以解释和计算效率低下。此外，这两种方法都不适合于特定分位数的子集选择。相反，我们从贝叶斯决策分析的角度出发，提出了线性分位估计、不确定性量化和子集选择的基本问题。对于任何贝叶斯回归模型，我们为每个基于模型的条件分位数推导出最佳和可解释的线性估计值和不确定性量化。我们的方法引入了一种分位数聚焦的方法。

    Quantile regression is a powerful tool for inferring how covariates affect specific percentiles of the response distribution. Existing methods either estimate conditional quantiles separately for each quantile of interest or estimate the entire conditional distribution using semi- or non-parametric models. The former often produce inadequate models for real data and do not share information across quantiles, while the latter are characterized by complex and constrained models that can be difficult to interpret and computationally inefficient. Further, neither approach is well-suited for quantile-specific subset selection. Instead, we pose the fundamental problems of linear quantile estimation, uncertainty quantification, and subset selection from a Bayesian decision analysis perspective. For any Bayesian regression model, we derive optimal and interpretable linear estimates and uncertainty quantification for each model-based conditional quantile. Our approach introduces a quantile-focu
    

