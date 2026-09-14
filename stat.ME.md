# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distributionally Robust Transfer Learning.](http://arxiv.org/abs/2309.06534) | 这篇论文介绍了一种分布鲁棒的迁移学习方法，通过优化一个不确定性集合内最具对抗性的损失来实现，该集合是由源分布的凸组合生成的目标人口集合，能够有效地将迁移学习和分布鲁棒的预测模型联系起来。 |

# 详细

[^1]: 分布鲁棒的迁移学习

    Distributionally Robust Transfer Learning. (arXiv:2309.06534v1 [cs.LG])

    [http://arxiv.org/abs/2309.06534](http://arxiv.org/abs/2309.06534)

    这篇论文介绍了一种分布鲁棒的迁移学习方法，通过优化一个不确定性集合内最具对抗性的损失来实现，该集合是由源分布的凸组合生成的目标人口集合，能够有效地将迁移学习和分布鲁棒的预测模型联系起来。

    

    许多现有的迁移学习方法依赖于利用与目标数据相似的源数据的信息。然而，这种方法经常忽视了可能存在于不同但潜在相关的辅助样本中的有价值的知识。当处理有限的目标数据和多样化的源模型时，我们的论文引入了一种新颖的方法，分布鲁棒迁移学习（TransDRO），它摆脱了严格的相似性约束。TransDRO通过在一个不确定性集合内优化最具对抗性的损失来设计，该集合定义为由源分布的凸组合生成的目标人口的集合，保证了对目标数据的出色预测性能。TransDRO有效地将迁移学习和分布鲁棒的预测模型联系起来。我们建立了TransDRO的可辨识性和其作为最接近源模型的加权平均值的解释。

    Many existing transfer learning methods rely on leveraging information from source data that closely resembles the target data. However, this approach often overlooks valuable knowledge that may be present in different yet potentially related auxiliary samples. When dealing with a limited amount of target data and a diverse range of source models, our paper introduces a novel approach, Distributionally Robust Optimization for Transfer Learning (TransDRO), that breaks free from strict similarity constraints. TransDRO is designed to optimize the most adversarial loss within an uncertainty set, defined as a collection of target populations generated as a convex combination of source distributions that guarantee excellent prediction performances for the target data. TransDRO effectively bridges the realms of transfer learning and distributional robustness prediction models. We establish the identifiability of TransDRO and its interpretation as a weighted average of source models closest to
    

