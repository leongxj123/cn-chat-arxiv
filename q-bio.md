# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management](https://arxiv.org/abs/2402.07949) | 通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。 |

# 详细

[^1]: 优化人工胰腺设计以改善糖尿病管理

    Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management

    [https://arxiv.org/abs/2402.07949](https://arxiv.org/abs/2402.07949)

    通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。

    

    糖尿病是一种慢性疾病，影响美国境内有3800万人，它会影响身体将食物转化为能量（即血糖）的能力。标准的治疗方法是通过使用人工胰腺，即持续胰岛素泵（基础注射），以及定期注射胰岛素（突发注射）来补充碳水化合物摄入量。治疗目标是将血糖保持在可接受范围的中心位置，通过持续血糖测量来进行衡量。次要目标是减少注射次数，因为对某些患者来说注射是不愉快且难以实施的。本研究使用神经进化来发现治疗的最佳策略。基于30天的治疗和单个患者的测量数据集，首先训练了随机森林来预测未来的血糖水平。然后通过进化了一个神经网络来指定碳水化合物摄入量、基础注射水平和突发注射。进化发现了一个帕累托前沿，减少了与目标值的偏差。

    Diabetes, a chronic condition that impairs how the body turns food into energy, i.e. blood glucose, affects 38 million people in the US alone. The standard treatment is to supplement carbohydrate intake with an artificial pancreas, i.e. a continuous insulin pump (basal shots), as well as occasional insulin injections (bolus shots). The goal of the treatment is to keep blood glucose at the center of an acceptable range, as measured through a continuous glucose meter. A secondary goal is to minimize injections, which are unpleasant and difficult for some patients to implement. In this study, neuroevolution was used to discover an optimal strategy for the treatment. Based on a dataset of 30 days of treatment and measurements of a single patient, a random forest was first trained to predict future glucose levels. A neural network was then evolved to prescribe carbohydrates, basal pumping levels, and bolus injections. Evolution discovered a Pareto front that reduced deviation from the targe
    

