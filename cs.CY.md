# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records](https://rss.arxiv.org/abs/2402.00955) | FairEHR-CLP是一个通用框架，利用对比学习，通过生成患者的合成对应来实现多样化的人口统计身份，并利用公平感知预测方法消除EHR中的社会偏见。 |

# 详细

[^1]: FairEHR-CLP：以对比学习实现的公平感知多模态电子健康记录中的临床预测

    FairEHR-CLP: Towards Fairness-Aware Clinical Predictions with Contrastive Learning in Multimodal Electronic Health Records

    [https://rss.arxiv.org/abs/2402.00955](https://rss.arxiv.org/abs/2402.00955)

    FairEHR-CLP是一个通用框架，利用对比学习，通过生成患者的合成对应来实现多样化的人口统计身份，并利用公平感知预测方法消除EHR中的社会偏见。

    

    在医疗保健领域中，确保预测模型的公平性至关重要。电子健康记录（EHR）已成为医疗决策的重要组成部分，然而现有的增强模型公平性的方法仅限于单模态数据，并未解决EHR中与人口统计因素交织在一起的多方面社会偏见。为了减轻这些偏见，我们提出了FairEHR-CLP：一种公平感知临床预测的通用框架，通过对比学习在EHR中进行操作。FairEHR-CLP通过两个阶段的过程操作，利用患者人口统计数据、纵向数据和临床记录。首先，为每个患者生成合成对应来实现多样化的人口统计身份，同时保留必要的健康信息。其次，公平感知预测利用对比学习将患者的表示在敏感属性上进行对齐，与具有softmax层的MLP分类器共同优化用于临床分类任务。

    In the high-stakes realm of healthcare, ensuring fairness in predictive models is crucial. Electronic Health Records (EHRs) have become integral to medical decision-making, yet existing methods for enhancing model fairness restrict themselves to unimodal data and fail to address the multifaceted social biases intertwined with demographic factors in EHRs. To mitigate these biases, we present FairEHR-CLP: a general framework for Fairness-aware Clinical Predictions with Contrastive Learning in EHRs. FairEHR-CLP operates through a two-stage process, utilizing patient demographics, longitudinal data, and clinical notes. First, synthetic counterparts are generated for each patient, allowing for diverse demographic identities while preserving essential health information. Second, fairness-aware predictions employ contrastive learning to align patient representations across sensitive attributes, jointly optimized with an MLP classifier with a softmax layer for clinical classification tasks. Ac
    

