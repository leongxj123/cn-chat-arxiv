# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Generative Models for Climbing Aircraft from Radar Data.](http://arxiv.org/abs/2309.14941) | 本文提出了一种利用雷达数据学习的生成模型，能够准确预测攀升飞机的轨迹，并通过学习修正推力的函数来提高预测准确性。该方法的优势包括：与标准模型相比，到达时间的预测误差减少了66.3%；生成的轨迹与测试数据相比更加真实；并且能够以最小的计算成本计算置信区间。 |

# 详细

[^1]: 从雷达数据学习攀升飞机的生成模型

    Learning Generative Models for Climbing Aircraft from Radar Data. (arXiv:2309.14941v1 [eess.SY])

    [http://arxiv.org/abs/2309.14941](http://arxiv.org/abs/2309.14941)

    本文提出了一种利用雷达数据学习的生成模型，能够准确预测攀升飞机的轨迹，并通过学习修正推力的函数来提高预测准确性。该方法的优势包括：与标准模型相比，到达时间的预测误差减少了66.3%；生成的轨迹与测试数据相比更加真实；并且能够以最小的计算成本计算置信区间。

    

    攀升飞机的准确轨迹预测受到机载设备操作的不确定性的影响，可能导致预测轨迹与观测轨迹之间存在显著的差异。本文提出了一种生成模型，通过从数据中学习修正推力的函数来丰富标准的飞机基础数据（BADA）模型。该方法具有三个特点：与BADA相比，到达时间的预测误差减少了66.3%；生成的轨迹与测试数据相比更加真实；并且能够以最小的计算成本计算置信区间。

    Accurate trajectory prediction (TP) for climbing aircraft is hampered by the presence of epistemic uncertainties concerning aircraft operation, which can lead to significant misspecification between predicted and observed trajectories. This paper proposes a generative model for climbing aircraft in which the standard Base of Aircraft Data (BADA) model is enriched by a functional correction to the thrust that is learned from data. The method offers three features: predictions of the arrival time with 66.3% less error when compared to BADA; generated trajectories that are realistic when compared to test data; and a means of computing confidence bounds for minimal computational cost.
    

