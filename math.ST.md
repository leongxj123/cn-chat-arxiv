# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^2] | [Large sample properties of GMM estimators under second-order identification.](http://arxiv.org/abs/2307.13475) | 本文提出了GMM估计器在二阶识别条件下的大样本性质，证明了估计器的收敛速率并给出了极限分布，但需要满足过识别条件。 |

# 详细

[^1]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^2]: GMM估计器在二阶识别下的大样本性质

    Large sample properties of GMM estimators under second-order identification. (arXiv:2307.13475v1 [econ.EM])

    [http://arxiv.org/abs/2307.13475](http://arxiv.org/abs/2307.13475)

    本文提出了GMM估计器在二阶识别条件下的大样本性质，证明了估计器的收敛速率并给出了极限分布，但需要满足过识别条件。

    

    本文翻译了Dovonon和Hall（2018）在全局识别参数向量{\phi}的p维情形中，当一阶识别条件失败但二阶识别条件成立时提出了GMM估计器的极限分布理论。他们假设一阶不识别是由于在真实值{\phi}_{0}处预期的Jacobian矩阵的秩为p-1，即存在秩缺失的情况。通过对模型重新参数化，使得Jacobian矩阵的最后一列为零，他们证明了前p-1个参数的GMM估计收敛速率为T^{-1/2}，剩下的参数{\phi}_{p}的GMM估计收敛速率为T^{-1/4}。他们还给出了T^{1/4}({\phi}_{p}-{\phi}_{0,p})的极限分布，但需要满足一个（不透明）条件，他们声称这个条件通常并不具限制性。然而，正如我们在本文中所展示的，他们的条件实际上只有在{\phi}过识别的情况下才满足。

    Dovonon and Hall (Journal of Econometrics, 2018) proposed a limiting distribution theory for GMM estimators for a p - dimensional globally identified parameter vector {\phi} when local identification conditions fail at first-order but hold at second-order. They assumed that the first-order underidentification is due to the expected Jacobian having rank p-1 at the true value {\phi}_{0}, i.e., having a rank deficiency of one. After reparametrizing the model such that the last column of the Jacobian vanishes, they showed that the GMM estimator of the first p-1 parameters converges at rate T^{-1/2} and the GMM estimator of the remaining parameter, {\phi}_{p}, converges at rate T^{-1/4}. They also provided a limiting distribution of T^{1/4}({\phi}_{p}-{\phi}_{0,p}) subject to a (non-transparent) condition which they claimed to be not restrictive in general. However, as we show in this paper, their condition is in fact only satisfied when {\phi} is overidentified and the limiting distributio
    

