# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fairness-aware Federated Minimax Optimization with Convergence Guarantee.](http://arxiv.org/abs/2307.04417) | 本文提出了一种名为FFALM的算法，通过施加公平约束和解决极小化极大回归问题，在联邦学习中解决了群体公平性问题。实验证明FFALM在处理严重统计异质性问题时具有良好的效果。 |
| [^2] | [Are demographically invariant models and representations in medical imaging fair?.](http://arxiv.org/abs/2305.01397) | 医学影像模型编码患者人口统计信息，引发有关潜在歧视的担忧。研究表明，不编码人口属性的模型容易损失预测性能，而考虑人口统计属性的反事实模型不变性存在复杂性。人口统计学编码可以被认为是优势。 |

# 详细

[^1]: 具有收敛保证的公正感知联邦极小化优化

    Fairness-aware Federated Minimax Optimization with Convergence Guarantee. (arXiv:2307.04417v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04417](http://arxiv.org/abs/2307.04417)

    本文提出了一种名为FFALM的算法，通过施加公平约束和解决极小化极大回归问题，在联邦学习中解决了群体公平性问题。实验证明FFALM在处理严重统计异质性问题时具有良好的效果。

    

    由于其保护隐私的特性，联邦学习 (FL) 吸引了相当多的关注。然而，管理用户数据的自由度不足可能导致群体公平性问题，即模型偏向于敏感因素诸如种族或性别。为了解决这个问题，本文提出了一种新颖的算法，名为带有增广拉格朗日方法的公平联邦平均法 (FFALM)，专门用于解决FL中的群体公平问题。具体来说，我们对训练目标施加了公平约束，并解决了受约束优化问题的极小化极大回归。然后，我们推导了FFALM的收敛速率的理论上界。通过在CelebA和UTKFace数据集中充分考虑严重统计异质性，实证结果表明了FFALM 在提高公平性方面的有效性。

    Federated learning (FL) has garnered considerable attention due to its privacy-preserving feature. Nonetheless, the lack of freedom in managing user data can lead to group fairness issues, where models are biased towards sensitive factors such as race or gender. To tackle this issue, this paper proposes a novel algorithm, fair federated averaging with augmented Lagrangian method (FFALM), designed explicitly to address group fairness issues in FL. Specifically, we impose a fairness constraint on the training objective and solve the minimax reformulation of the constrained optimization problem. Then, we derive the theoretical upper bound for the convergence rate of FFALM. The effectiveness of FFALM in improving fairness is shown empirically on CelebA and UTKFace datasets in the presence of severe statistical heterogeneity.
    
[^2]: 医学影像中的人口统计学不变模型和表示是否公平？

    Are demographically invariant models and representations in medical imaging fair?. (arXiv:2305.01397v1 [cs.LG])

    [http://arxiv.org/abs/2305.01397](http://arxiv.org/abs/2305.01397)

    医学影像模型编码患者人口统计信息，引发有关潜在歧视的担忧。研究表明，不编码人口属性的模型容易损失预测性能，而考虑人口统计属性的反事实模型不变性存在复杂性。人口统计学编码可以被认为是优势。

    

    研究表明，医学成像模型在其潜在表示中编码了有关患者人口统计学信息（年龄、种族、性别），这引发了有关其潜在歧视的担忧。在这里，我们询问是否可行和值得训练不编码人口属性的模型。我们考虑不同类型的与人口统计学属性的不变性，即边际、类条件和反事实模型不变性，并说明它们与算法公平的标准概念的等价性。根据现有理论，我们发现边际和类条件的不变性可被认为是实现某些公平概念的过度限制方法，导致显著的预测性能损失。关于反事实模型不变性，我们注意到对于人口统计学属性，定义医学图像反事实存在复杂性。最后，我们认为人口统计学编码甚至可以被认为是优势。

    Medical imaging models have been shown to encode information about patient demographics (age, race, sex) in their latent representation, raising concerns about their potential for discrimination. Here, we ask whether it is feasible and desirable to train models that do not encode demographic attributes. We consider different types of invariance with respect to demographic attributes marginal, class-conditional, and counterfactual model invariance - and lay out their equivalence to standard notions of algorithmic fairness. Drawing on existing theory, we find that marginal and class-conditional invariance can be considered overly restrictive approaches for achieving certain fairness notions, resulting in significant predictive performance losses. Concerning counterfactual model invariance, we note that defining medical image counterfactuals with respect to demographic attributes is fraught with complexities. Finally, we posit that demographic encoding may even be considered advantageou
    

