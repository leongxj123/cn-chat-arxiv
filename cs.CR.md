# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^2] | [Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats.](http://arxiv.org/abs/2401.10375) | 本文研究基于基础模型集成的联邦学习在敌对威胁下的漏洞，提出了一种新的攻击策略，揭示了该模型在不同配置的联邦学习下对敌对威胁的高敏感性。 |

# 详细

[^1]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^2]: 基于基础模型集成的联邦学习在敌对威胁下的漏洞

    Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats. (arXiv:2401.10375v1 [cs.CR])

    [http://arxiv.org/abs/2401.10375](http://arxiv.org/abs/2401.10375)

    本文研究基于基础模型集成的联邦学习在敌对威胁下的漏洞，提出了一种新的攻击策略，揭示了该模型在不同配置的联邦学习下对敌对威胁的高敏感性。

    

    联邦学习是解决与数据隐私和安全相关的机器学习的重要问题，但在某些情况下存在数据不足和不平衡问题。基础模型的出现为现有联邦学习框架的局限性提供了潜在的解决方案，例如通过生成合成数据进行模型初始化。然而，由于基础模型的内在安全性问题，将基础模型集成到联邦学习中可能引入新的风险，这方面的研究尚属未开发。为了填补这一空白，我们首次研究基于基础模型集成的联邦学习在敌对威胁下的漏洞。基于基础模型集成的联邦学习的统一框架，我们引入了一种新的攻击策略，利用基础模型的安全性问题来破坏联邦学习客户端模型。通过在图像和文本领域中使用知名模型和基准数据集进行广泛实验，我们揭示了基于基础模型集成的联邦学习在不同配置的联邦学习下对这种新威胁的高敏感性。

    Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we f
    

