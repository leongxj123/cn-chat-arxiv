# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Interpretable Generalization Mechanism for Accurately Detecting Anomaly and Identifying Networking Intrusion Techniques](https://arxiv.org/abs/2403.07959) | IG是一个可解释泛化机制，能够准确区分正常和异常的网络流量，并揭示复杂的入侵路径，为网络安全取证提供重要见解。 |
| [^2] | [Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics.](http://arxiv.org/abs/2401.05146) | 这篇综述论文介绍了联邦遗忘的概念和挑战，以及解决这些问题的方法和设计准则，旨在为联邦学习中保护用户隐私和防止恶意攻击提供解决方案。 |

# 详细

[^1]: 一个用于准确检测异常并识别网络入侵技术的可解释泛化机制

    An Interpretable Generalization Mechanism for Accurately Detecting Anomaly and Identifying Networking Intrusion Techniques

    [https://arxiv.org/abs/2403.07959](https://arxiv.org/abs/2403.07959)

    IG是一个可解释泛化机制，能够准确区分正常和异常的网络流量，并揭示复杂的入侵路径，为网络安全取证提供重要见解。

    

    最近入侵检测系统（IDS）中整合可解释人工智能（XAI）方法的发展，通过精确的特征选择，显著提升了系统性能。然而，对网络攻击的彻底理解要求IDS内在可解释的决策过程。本文介绍了“可解释泛化机制”（IG），旨在彻底改变IDS的能力。IG能够识别连贯模式，使其能够解释区分正常和异常的网络流量。此外，连贯模式的综合揭示复杂的入侵路径，为网络安全取证提供了必要的见解。通过对真实数据集NSL-KDD、UNSW-NB15和UKM-IDS20的实验，IG即使在较低的训练-测试比率下也能准确。在NSL-KDD数据集中，当训练-测试比率为10%-90%时，IG实现的Precision（PRE）=0.93、Recall（REC）=0.94和Area Under Curve（AUC）=0.94；PRE=0.98...

    arXiv:2403.07959v1 Announce Type: cross  Abstract: Recent advancements in Intrusion Detection Systems (IDS), integrating Explainable AI (XAI) methodologies, have led to notable improvements in system performance via precise feature selection. However, a thorough understanding of cyber-attacks requires inherently explainable decision-making processes within IDS. In this paper, we present the Interpretable Generalization Mechanism (IG), poised to revolutionize IDS capabilities. IG discerns coherent patterns, making it interpretable in distinguishing between normal and anomalous network traffic. Further, the synthesis of coherent patterns sheds light on intricate intrusion pathways, providing essential insights for cybersecurity forensics. By experiments with real-world datasets NSL-KDD, UNSW-NB15, and UKM-IDS20, IG is accurate even at a low ratio of training-to-test. With 10%-to-90%, IG achieves Precision (PRE)=0.93, Recall (REC)=0.94, and Area Under Curve (AUC)=0.94 in NSL-KDD; PRE=0.98
    
[^2]: 联邦遗忘：方法、设计准则和评估指标的综述

    Federated Unlearning: A Survey on Methods, Design Guidelines, and Evaluation Metrics. (arXiv:2401.05146v1 [cs.LG])

    [http://arxiv.org/abs/2401.05146](http://arxiv.org/abs/2401.05146)

    这篇综述论文介绍了联邦遗忘的概念和挑战，以及解决这些问题的方法和设计准则，旨在为联邦学习中保护用户隐私和防止恶意攻击提供解决方案。

    

    联邦学习使得多个参与方能够协同训练一个机器学习模型，通过保留数据在本地存储，从而维护了用户和机构的隐私。与集中化原始数据不同，联邦学习通过交换本地优化的模型参数来逐步构建全局模型。尽管联邦学习更加符合新兴规定，如欧洲通用数据保护条例（GDPR），但在此背景下确保遗忘权——允许联邦学习参与方从学习的模型中删除他们的数据贡献仍然不明确。此外，人们认识到恶意客户端可能通过更新将后门注入全局模型，例如对特制数据示例进行错误预测。因此，需要机制来确保个人有可能在聚合后移除他们的数据并清除恶意贡献，而不损害已获得的"全

    Federated Learning (FL) enables collaborative training of a Machine Learning (ML) model across multiple parties, facilitating the preservation of users' and institutions' privacy by keeping data stored locally. Instead of centralizing raw data, FL exchanges locally refined model parameters to build a global model incrementally. While FL is more compliant with emerging regulations such as the European General Data Protection Regulation (GDPR), ensuring the right to be forgotten in this context - allowing FL participants to remove their data contributions from the learned model - remains unclear. In addition, it is recognized that malicious clients may inject backdoors into the global model through updates, e.g. to generate mispredictions on specially crafted data examples. Consequently, there is the need for mechanisms that can guarantee individuals the possibility to remove their data and erase malicious contributions even after aggregation, without compromising the already acquired "g
    

