# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unveiling the Blind Spots: A Critical Examination of Fairness in Autonomous Driving Systems](https://arxiv.org/abs/2308.02935) | 该研究对当前深度学习行人检测器的公平性进行了全面评估，发现了与年龄相关的重要公平性问题。 |
| [^2] | [Fairness-aware Federated Minimax Optimization with Convergence Guarantee.](http://arxiv.org/abs/2307.04417) | 本文提出了一种名为FFALM的算法，通过施加公平约束和解决极小化极大回归问题，在联邦学习中解决了群体公平性问题。实验证明FFALM在处理严重统计异质性问题时具有良好的效果。 |

# 详细

[^1]: 揭示盲点：对自动驾驶系统中公平性的关键审查

    Unveiling the Blind Spots: A Critical Examination of Fairness in Autonomous Driving Systems

    [https://arxiv.org/abs/2308.02935](https://arxiv.org/abs/2308.02935)

    该研究对当前深度学习行人检测器的公平性进行了全面评估，发现了与年龄相关的重要公平性问题。

    

    自主驾驶系统已经扩展了智能车辆物联网的范围，并成为Web生态系统的重要组成部分。类似于传统的基于Web的应用程序，公平性对于确保自动驾驶系统的高质量是一个重要方面，特别是在其中的行人检测器的背景下。然而，目前关于当前基于深度学习（DL）的行人检测器公平性的综合评估在文献中尚未出现。为了填补这一空白，我们在大规模真实世界数据集上评估了八种被广泛探索的DL行人检测器在人口统计学群体之间的表现。为了实现彻底的公平性评估，我们为数据集提供了广泛的注释，共涉及8,311张图像，16,070个性别标签，20,115个年龄标签和3,513个肤色标签。我们的研究发现了与年龄相关的重要公平性问题。

    arXiv:2308.02935v2 Announce Type: replace-cross  Abstract: Autonomous driving systems have extended the spectrum of Web of Things for intelligent vehicles and have become an important component of the Web ecosystem. Similar to traditional Web-based applications, fairness is an essential aspect for ensuring the high quality of autonomous driving systems, particularly in the context of pedestrian detectors within them. However, there is an absence in the literature of a comprehensive assessment of the fairness of current Deep Learning (DL)-based pedestrian detectors. To fill the gap, we evaluate eight widely-explored DL-based pedestrian detectors across demographic groups on large-scale real-world datasets. To enable a thorough fairness evaluation, we provide extensive annotations for the datasets, resulting in 8,311 images with 16,070 gender labels, 20,115 age labels, and 3,513 skin tone labels. Our findings reveal significant fairness issues related to age. The undetected proportions f
    
[^2]: 具有收敛保证的公正感知联邦极小化优化

    Fairness-aware Federated Minimax Optimization with Convergence Guarantee. (arXiv:2307.04417v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04417](http://arxiv.org/abs/2307.04417)

    本文提出了一种名为FFALM的算法，通过施加公平约束和解决极小化极大回归问题，在联邦学习中解决了群体公平性问题。实验证明FFALM在处理严重统计异质性问题时具有良好的效果。

    

    由于其保护隐私的特性，联邦学习 (FL) 吸引了相当多的关注。然而，管理用户数据的自由度不足可能导致群体公平性问题，即模型偏向于敏感因素诸如种族或性别。为了解决这个问题，本文提出了一种新颖的算法，名为带有增广拉格朗日方法的公平联邦平均法 (FFALM)，专门用于解决FL中的群体公平问题。具体来说，我们对训练目标施加了公平约束，并解决了受约束优化问题的极小化极大回归。然后，我们推导了FFALM的收敛速率的理论上界。通过在CelebA和UTKFace数据集中充分考虑严重统计异质性，实证结果表明了FFALM 在提高公平性方面的有效性。

    Federated learning (FL) has garnered considerable attention due to its privacy-preserving feature. Nonetheless, the lack of freedom in managing user data can lead to group fairness issues, where models are biased towards sensitive factors such as race or gender. To tackle this issue, this paper proposes a novel algorithm, fair federated averaging with augmented Lagrangian method (FFALM), designed explicitly to address group fairness issues in FL. Specifically, we impose a fairness constraint on the training objective and solve the minimax reformulation of the constrained optimization problem. Then, we derive the theoretical upper bound for the convergence rate of FFALM. The effectiveness of FFALM in improving fairness is shown empirically on CelebA and UTKFace datasets in the presence of severe statistical heterogeneity.
    

