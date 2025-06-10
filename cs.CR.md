# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective](https://arxiv.org/abs/2402.10686) | 通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险 |
| [^2] | [SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models.](http://arxiv.org/abs/2401.00793) | SecFormer是一个优化框架，旨在实现Transformer模型的快速准确隐私保护推理。通过消除高成本的指数和线性操作，SecFormer能够有效解决在大型语言模型中应用SMPC时的性能问题。 |

# 详细

[^1]: 不确定性、校准和成员推理攻击：信息论视角

    Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective

    [https://arxiv.org/abs/2402.10686](https://arxiv.org/abs/2402.10686)

    通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险

    

    在成员推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否被用于训练目标模型。在本文中，我们在一个信息理论框架内分析了最先进的似然比攻击（LiRA）的性能，这个框架可以允许研究真实数据生成过程中的不确定性的影响，由有限训练数据集引起的认知不确定性以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收到的信息逐渐减少：置信向量（CV）披露，其中输出概率向量被发布；真实标签置信度（TLC）披露，其中只有模型分配给真实标签的概率是可用的；以及决策集（DS）披露。

    arXiv:2402.10686v1 Announce Type: cross  Abstract: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in 
    
[^2]: SecFormer：面向大型语言模型的快速准确隐私保护推理

    SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models. (arXiv:2401.00793v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00793](http://arxiv.org/abs/2401.00793)

    SecFormer是一个优化框架，旨在实现Transformer模型的快速准确隐私保护推理。通过消除高成本的指数和线性操作，SecFormer能够有效解决在大型语言模型中应用SMPC时的性能问题。

    

    随着在云平台上部署大型语言模型以提供推理服务的使用增加，隐私问题日益加剧，尤其是涉及投资计划和银行账户等敏感数据。安全多方计算（SMPC）被视为保护推理数据和模型参数隐私的一种有前途的解决方案。然而，SMPC在大型语言模型（特别是基于Transformer架构的模型）的隐私保护推理中的应用往往会导致显著的减速或性能下降。这主要是由于Transformer架构中的众多非线性操作不适合SMPC，并且难以有效规避或优化。为了解决这个问题，我们引入了一个先进的优化框架，称为SecFormer，以实现Transformer模型的快速准确隐私保护推理。通过实施模型设计优化，我们成功消除了高成本的指数和线性操作，并取得了良好的性能。

    With the growing use of large language models hosted on cloud platforms to offer inference services, privacy concerns are escalating, especially concerning sensitive data like investment plans and bank account details. Secure Multi-Party Computing (SMPC) emerges as a promising solution to protect the privacy of inference data and model parameters. However, the application of SMPC in Privacy-Preserving Inference (PPI) for large language models, particularly those based on the Transformer architecture, often leads to considerable slowdowns or declines in performance. This is largely due to the multitude of nonlinear operations in the Transformer architecture, which are not well-suited to SMPC and difficult to circumvent or optimize effectively. To address this concern, we introduce an advanced optimization framework called SecFormer, to achieve fast and accurate PPI for Transformer models. By implementing model design optimization, we successfully eliminate the high-cost exponential and 
    

