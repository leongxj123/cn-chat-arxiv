# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards A Robust Group-level Emotion Recognition via Uncertainty-Aware Learning.](http://arxiv.org/abs/2310.04306) | 本文提出了一种考虑不确定性的学习方法用于群体级情绪识别。通过模型化每个个体的不确定性，利用随机嵌入来代替确定性的点嵌入。这种表示能够捕捉概率和在推断阶段产生多样的预测。 |

# 详细

[^1]: 通过考虑不确定性的学习方法实现鲁棒的群体级情绪识别

    Towards A Robust Group-level Emotion Recognition via Uncertainty-Aware Learning. (arXiv:2310.04306v1 [cs.CV])

    [http://arxiv.org/abs/2310.04306](http://arxiv.org/abs/2310.04306)

    本文提出了一种考虑不确定性的学习方法用于群体级情绪识别。通过模型化每个个体的不确定性，利用随机嵌入来代替确定性的点嵌入。这种表示能够捕捉概率和在推断阶段产生多样的预测。

    

    群体级情绪识别是人类行为分析中不可分割的一部分，旨在识别多人场景中的整体情绪。然而，现有方法致力于整合不同的情绪线索，而忽视了在无约束环境下存在的团体内拥挤和遮挡等固有不确定性。此外，由于仅有群体级标签可用，在一个群体中个体之间的不一致情绪预测会混淆网络。在本文中，我们提出了一种考虑不确定性的学习方法，为群体级情绪识别提取更加鲁棒的表示。通过明确地建模每个个体的不确定性，我们利用高斯分布中的随机嵌入来代替确定性的点嵌入。这种表示捕捉了不同情绪的概率，并通过这种随机性在推断阶段产生多样的预测。

    Group-level emotion recognition (GER) is an inseparable part of human behavior analysis, aiming to recognize an overall emotion in a multi-person scene. However, the existing methods are devoted to combing diverse emotion cues while ignoring the inherent uncertainties under unconstrained environments, such as congestion and occlusion occurring within a group. Additionally, since only group-level labels are available, inconsistent emotion predictions among individuals in one group can confuse the network. In this paper, we propose an uncertainty-aware learning (UAL) method to extract more robust representations for GER. By explicitly modeling the uncertainty of each individual, we utilize stochastic embedding drawn from a Gaussian distribution instead of deterministic point embedding. This representation captures the probabilities of different emotions and generates diverse predictions through this stochasticity during the inference stage. Furthermore, uncertainty-sensitive scores are a
    

