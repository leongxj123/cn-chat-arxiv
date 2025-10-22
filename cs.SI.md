# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis.](http://arxiv.org/abs/2311.00164) | 该论文构建了一个大规模的道路交通事故记录数据集，并使用该数据集评估了现有的深度学习方法在预测事故发生方面的准确性。研究发现，图神经网络GraphSAGE能够准确预测道路上的事故数量，并判断事故是否会发生。 |

# 详细

[^1]: 道路安全建模的图神经网络：用于事故分析的数据集和评估

    Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis. (arXiv:2311.00164v1 [cs.SI])

    [http://arxiv.org/abs/2311.00164](http://arxiv.org/abs/2311.00164)

    该论文构建了一个大规模的道路交通事故记录数据集，并使用该数据集评估了现有的深度学习方法在预测事故发生方面的准确性。研究发现，图神经网络GraphSAGE能够准确预测道路上的事故数量，并判断事故是否会发生。

    

    我们考虑基于道路网络连接和交通流量的道路网络上的交通事故分析问题。以往的工作使用历史记录设计了各种深度学习方法来预测交通事故的发生。然而，现有方法的准确性缺乏共识，并且一个基本问题是缺乏公共事故数据集进行全面评估。本文构建了一个大规模的、统一的道路交通事故记录数据集，包括来自美国各州官方报告的900万条记录，以及道路网络和交通流量报告。利用这个新数据集，我们评估了现有的深度学习方法来预测道路网络上的事故发生。我们的主要发现是，像GraphSAGE这样的图神经网络可以准确预测道路上的事故数量，平均绝对误差不超过实际数目的22%，并能够判断事故是否会发生。

    We consider the problem of traffic accident analysis on a road network based on road network connections and traffic volume. Previous works have designed various deep-learning methods using historical records to predict traffic accident occurrences. However, there is a lack of consensus on how accurate existing methods are, and a fundamental issue is the lack of public accident datasets for comprehensive evaluations. This paper constructs a large-scale, unified dataset of traffic accident records from official reports of various states in the US, totaling 9 million records, accompanied by road networks and traffic volume reports. Using this new dataset, we evaluate existing deep-learning methods for predicting the occurrence of accidents on road networks. Our main finding is that graph neural networks such as GraphSAGE can accurately predict the number of accidents on roads with less than 22% mean absolute error (relative to the actual count) and whether an accident will occur or not w
    

