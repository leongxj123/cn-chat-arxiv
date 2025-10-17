# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space.](http://arxiv.org/abs/2312.17300) | 本研究提出了一种使用多任务学习的两阶段表示学习技术，通过培养潜在空间中的特征，包括本地和跨领域特征，以增强对未知分布领域的泛化效果。此外，通过最小化先验和潜在空间之间的互信息来分离潜在空间，并且在多个网络安全数据集上评估了模型的效能。 |
| [^2] | [Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning.](http://arxiv.org/abs/2306.05494) | 本文对于基于机器学习的网络入侵检测系统(NIDS)的对抗性攻击进行了分类，同时探究了持续再训练对NIDS对抗性攻击的影响。实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。 |

# 详细

[^1]: 在潜在空间中通过领域不变表示学习改善入侵检测

    Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space. (arXiv:2312.17300v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2312.17300](http://arxiv.org/abs/2312.17300)

    本研究提出了一种使用多任务学习的两阶段表示学习技术，通过培养潜在空间中的特征，包括本地和跨领域特征，以增强对未知分布领域的泛化效果。此外，通过最小化先验和潜在空间之间的互信息来分离潜在空间，并且在多个网络安全数据集上评估了模型的效能。

    

    领域泛化聚焦于利用来自具有丰富训练数据和标签的多个相关领域的知识，增强对未知分布（IN）和超出分布（OOD）领域的推理。在我们的研究中，我们引入了一种两阶段表示学习技术，使用多任务学习。这种方法旨在从跨越多个领域的特征中培养一个潜在空间，包括本地和跨领域，以增强对IN和OOD领域的泛化。此外，我们尝试通过最小化先验与潜在空间之间的互信息来分离潜在空间，有效消除虚假特征相关性。综合而言，联合优化将促进领域不变特征学习。我们使用标准分类指标评估模型在多个网络安全数据集上的效能，对比了现代领域泛化方法的结果。

    Domain generalization focuses on leveraging knowledge from multiple related domains with ample training data and labels to enhance inference on unseen in-distribution (IN) and out-of-distribution (OOD) domains. In our study, we introduce a two-phase representation learning technique using multi-task learning. This approach aims to cultivate a latent space from features spanning multiple domains, encompassing both native and cross-domains, to amplify generalization to IN and OOD territories. Additionally, we attempt to disentangle the latent space by minimizing the mutual information between the prior and latent space, effectively de-correlating spurious feature correlations. Collectively, the joint optimization will facilitate domain-invariant feature learning. We assess the model's efficacy across multiple cybersecurity datasets, using standard classification metrics on both unseen IN and OOD sets, and juxtapose the results with contemporary domain generalization methods.
    
[^2]: 神经网络中对抗性漏洞攻击的实用性测试：动态学习的影响

    Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning. (arXiv:2306.05494v1 [cs.CR])

    [http://arxiv.org/abs/2306.05494](http://arxiv.org/abs/2306.05494)

    本文对于基于机器学习的网络入侵检测系统(NIDS)的对抗性攻击进行了分类，同时探究了持续再训练对NIDS对抗性攻击的影响。实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。

    

    机器学习被广泛应用于网络入侵检测系统(NIDS)中，由于其自动化的特性和在处理和分类大量数据上的高精度。但机器学习存在缺陷，其中最大的问题之一是对抗性攻击，其目的是使机器学习模型产生错误的预测。本文提出了两个独特的贡献：对抗性攻击对基于机器学习的NIDS实用性问题的分类和对持续训练对NIDS对抗性攻击的影响进行了研究。我们的实验表明，即使没有对抗性训练，持续再训练也可以减少对抗性攻击的影响。虽然对抗性攻击可能会危及基于机器学习的NIDS，但持续再训练可带来一定的缓解效果。

    Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy in processing and classifying large volumes of data. However, ML has been found to have several flaws, on top of them are adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the practicality of such attacks against ML-based network security entities, especially NIDS.  This paper presents two distinct contributions: a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS and an investigation of the impact of continuous training on adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effect of adversarial attacks. While adversarial attacks can harm ML-based NIDSs, ou
    

