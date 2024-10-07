# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning.](http://arxiv.org/abs/2310.04055) | 本文提出了一种基于零知识证明的联邦学习异常检测方法，实现了在实际系统中检测和消除恶意客户端模型的能力。 |

# 详细

[^1]: 把坏人踢出去！基于零知识证明的联邦学习异常检测

    Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning. (arXiv:2310.04055v1 [cs.CR])

    [http://arxiv.org/abs/2310.04055](http://arxiv.org/abs/2310.04055)

    本文提出了一种基于零知识证明的联邦学习异常检测方法，实现了在实际系统中检测和消除恶意客户端模型的能力。

    

    联邦学习系统容易受到恶意客户端的攻击，他们通过提交篡改的本地模型来达到对抗目标，比如阻止全局模型的收敛或者导致全局模型对某些数据进行错误分类。许多现有的防御机制在实际联邦学习系统中不可行，因为它们需要先知道恶意客户端的数量，或者依赖重新加权或修改提交的方式。这是因为攻击者通常不会在攻击之前宣布他们的意图，而重新加权可能会改变聚合结果，即使没有攻击。为了解决这些在实际联邦学习系统中的挑战，本文引入了一种最尖端的异常检测方法，具有以下特点：i）仅在发生攻击时检测攻击的发生并进行防御操作；ii）一旦发生攻击，进一步检测恶意客户端模型并将其消除，而不会对正常模型造成伤害；iii）确保

    Federated learning (FL) systems are vulnerable to malicious clients that submit poisoned local models to achieve their adversarial goals, such as preventing the convergence of the global model or inducing the global model to misclassify some data. Many existing defense mechanisms are impractical in real-world FL systems, as they require prior knowledge of the number of malicious clients or rely on re-weighting or modifying submissions. This is because adversaries typically do not announce their intentions before attacking, and re-weighting might change aggregation results even in the absence of attacks. To address these challenges in real FL systems, this paper introduces a cutting-edge anomaly detection approach with the following features: i) Detecting the occurrence of attacks and performing defense operations only when attacks happen; ii) Upon the occurrence of an attack, further detecting the malicious client models and eliminating them without harming the benign ones; iii) Ensuri
    

