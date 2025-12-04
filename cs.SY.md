# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tackling Missing Values in Probabilistic Wind Power Forecasting: A Generative Approach](https://arxiv.org/abs/2403.03631) | 本文提出了一种新的概率风力发电预测方法，通过生成模型估计特征和目标的联合分布，同时预测所有未知值，避免了预处理环节，在连续排名概率得分方面比传统方法表现更优。 |

# 详细

[^1]: 处理概率风力发电预测中的缺失值：一种生成方法

    Tackling Missing Values in Probabilistic Wind Power Forecasting: A Generative Approach

    [https://arxiv.org/abs/2403.03631](https://arxiv.org/abs/2403.03631)

    本文提出了一种新的概率风力发电预测方法，通过生成模型估计特征和目标的联合分布，同时预测所有未知值，避免了预处理环节，在连续排名概率得分方面比传统方法表现更优。

    

    机器学习技术已成功应用于概率风力发电预测。然而，由于传感器故障等原因导致数据集中存在缺失值的问题长期以来被忽视。尽管通常在模型估计和预测之前通过插补缺失值来解决这个问题是很自然的，但我们建议将缺失值和预测目标视为同等重要，并基于观测值同时预测所有未知值。本文通过基于生成模型估计特征和目标的联合分布，提出了一种有效的概率预测方法。这种方法无需预处理，避免引入潜在的错误。与传统的“插补，然后预测”流程相比，该方法在连续排名概率得分方面表现更好。

    arXiv:2403.03631v1 Announce Type: new  Abstract: Machine learning techniques have been successfully used in probabilistic wind power forecasting. However, the issue of missing values within datasets due to sensor failure, for instance, has been overlooked for a long time. Although it is natural to consider addressing this issue by imputing missing values before model estimation and forecasting, we suggest treating missing values and forecasting targets indifferently and predicting all unknown values simultaneously based on observations. In this paper, we offer an efficient probabilistic forecasting approach by estimating the joint distribution of features and targets based on a generative model. It is free of preprocessing, and thus avoids introducing potential errors. Compared with the traditional "impute, then predict" pipeline, the proposed approach achieves better performance in terms of continuous ranked probability score.
    

