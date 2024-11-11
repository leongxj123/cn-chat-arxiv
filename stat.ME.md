# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semi-Supervised Health Index Monitoring with Feature Generation and Fusion](https://arxiv.org/abs/2312.02867) | 通过深度半监督异常检测（DeepSAD）方法进行健康指数构建，并提出了多样性损失来丰富条件指标。 |
| [^2] | [Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift.](http://arxiv.org/abs/2302.10160) | 该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。 |

# 详细

[^1]: 使用特征生成和融合的半监督健康指数监测

    Semi-Supervised Health Index Monitoring with Feature Generation and Fusion

    [https://arxiv.org/abs/2312.02867](https://arxiv.org/abs/2312.02867)

    通过深度半监督异常检测（DeepSAD）方法进行健康指数构建，并提出了多样性损失来丰富条件指标。

    

    健康指数（HI）对于评估系统健康状态至关重要，有助于识别异常，并预测对高安全性和可靠性要求高的系统的剩余使用寿命。在实现高精度的同时降低成本，紧密监测至关重要。在现实应用中获取HI标签往往成本高昂，需要连续、精确的健康测量。因此，利用可能提供潜在机器磨损状态指示的“运行至故障”数据集，更方便采用半监督工具构建HI。

    arXiv:2312.02867v2 Announce Type: replace  Abstract: The Health Index (HI) is crucial for evaluating system health, aiding tasks like anomaly detection and predicting remaining useful life for systems demanding high safety and reliability. Tight monitoring is crucial for achieving high precision at a lower cost. Obtaining HI labels in real-world applications is often cost-prohibitive, requiring continuous, precise health measurements. Therefore, it is more convenient to leverage run-to failure datasets that may provide potential indications of machine wear condition, making it necessary to apply semi-supervised tools for HI construction. In this study, we adapt the Deep Semi-supervised Anomaly Detection (DeepSAD) method for HI construction. We use the DeepSAD embedding as a condition indicators to address interpretability challenges and sensitivity to system-specific factors. Then, we introduce a diversity loss to enrich condition indicators. We employ an alternating projection algorit
    
[^2]: 核岭回归下伪标签的协变量转移策略

    Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift. (arXiv:2302.10160v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.10160](http://arxiv.org/abs/2302.10160)

    该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。

    

    我们提出并分析了一种基于协变量转移的核岭回归方法。我们的目标是在目标分布上学习一个均方误差最小的回归函数，基于从目标分布采样的未标记数据和可能具有不同特征分布的已标记数据。我们将已标记数据分成两个子集，并分别进行核岭回归，以获得候选模型集合和一个填充模型。我们使用后者填充缺失的标签，然后相应地选择最佳的候选模型。我们的非渐近性过量风险界表明，在相当一般的情况下，我们的估计器能够适应目标分布以及协变量转移的结构。它能够实现渐近正态误差率直到对数因子的最小极限优化。在模型选择中使用伪标签不会产生主要负面影响。

    We develop and analyze a principled approach to kernel ridge regression under covariate shift. The goal is to learn a regression function with small mean squared error over a target distribution, based on unlabeled data from there and labeled data that may have a different feature distribution. We propose to split the labeled data into two subsets and conduct kernel ridge regression on them separately to obtain a collection of candidate models and an imputation model. We use the latter to fill the missing labels and then select the best candidate model accordingly. Our non-asymptotic excess risk bounds show that in quite general scenarios, our estimator adapts to the structure of the target distribution as well as the covariate shift. It achieves the minimax optimal error rate up to a logarithmic factor. The use of pseudo-labels in model selection does not have major negative impacts.
    

