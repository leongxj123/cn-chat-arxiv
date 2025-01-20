# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Treatment Effect Estimation with Observational Network Data using Machine Learning.](http://arxiv.org/abs/2206.14591) | 该论文开发了增广逆概率加权（AIPW）方法，用于使用观测网络数据估计和推断具有溢出效应的治疗的直接效应。方法使用机器学习和样本分割，得到收敛速度较快且服从高斯分布的半参数治疗效果估计器。研究发现，在考虑学生社交网络的情况下，学习时间对考试成绩有影响。 |

# 详细

[^1]: 使用机器学习处理观测网络数据的治疗效果估计

    Treatment Effect Estimation with Observational Network Data using Machine Learning. (arXiv:2206.14591v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2206.14591](http://arxiv.org/abs/2206.14591)

    该论文开发了增广逆概率加权（AIPW）方法，用于使用观测网络数据估计和推断具有溢出效应的治疗的直接效应。方法使用机器学习和样本分割，得到收敛速度较快且服从高斯分布的半参数治疗效果估计器。研究发现，在考虑学生社交网络的情况下，学习时间对考试成绩有影响。

    

    因果推断方法通常假设独立单元来进行治疗效果估计。然而，这种假设经常是有问题的，因为单元之间可能会相互作用，导致单元之间的溢出效应。我们开发了增广逆概率加权（AIPW）方法，用于使用具有溢出效应的单个（社交）网络的观测数据对治疗的直接效应进行估计和推断。我们使用基于插件的机器学习和样本分割方法，得到一个半参数的治疗效果估计器，其渐近收敛于参数速率，并且在渐近情况下服从高斯分布。我们将AIPW方法应用于瑞士学生人生研究数据，以研究学习时间对考试成绩的影响，考虑到学生的社交网络。

    Causal inference methods for treatment effect estimation usually assume independent units. However, this assumption is often questionable because units may interact, resulting in spillover effects between units. We develop augmented inverse probability weighting (AIPW) for estimation and inference of the direct effect of the treatment with observational data from a single (social) network with spillover effects. We use plugin machine learning and sample splitting to obtain a semiparametric treatment effect estimator that converges at the parametric rate and asymptotically follows a Gaussian distribution. We apply our AIPW method to the Swiss StudentLife Study data to investigate the effect of hours spent studying on exam performance accounting for the students' social network.
    

