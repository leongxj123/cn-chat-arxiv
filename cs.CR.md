# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Secure Aggregation is Not Private Against Membership Inference Attacks](https://arxiv.org/abs/2403.17775) | 本文探讨了安全聚合在隐私方面的问题，揭示了其在面对成员推断攻击时并不具备足够的私密性。 |
| [^2] | [A Survey on Consumer IoT Traffic: Security and Privacy](https://arxiv.org/abs/2403.16149) | 本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。 |
| [^3] | [Pandora's White-Box: Increased Training Data Leakage in Open LLMs](https://arxiv.org/abs/2402.17012) | 本文对开源大型语言模型（LLMs）进行了隐私攻击研究，提出了首个能同时实现高真正率和低误分类率的预训练LLMs会员推理攻击（MIAs），以及展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。 |

# 详细

[^1]: 安全聚合在面对成员推断攻击时并非私密的

    Secure Aggregation is Not Private Against Membership Inference Attacks

    [https://arxiv.org/abs/2403.17775](https://arxiv.org/abs/2403.17775)

    本文探讨了安全聚合在隐私方面的问题，揭示了其在面对成员推断攻击时并不具备足够的私密性。

    

    安全聚合（SecAgg）是联邦学习中常用的隐私增强机制，仅允许服务器访问模型更新的聚合结果，同时保护个体更新的机密性。尽管有关SecAgg保护隐私能力的广泛声明，但缺乏对其隐私性的正式分析，因此这些假设是不合理的。本文通过将SecAgg视为每个局部更新的局部差分隐私（LDP）机制，深入探讨了SecAgg的隐私影响。我们设计了一种简单攻击方式，其中对手服务器试图在SecAgg下的联邦学习的单一训练轮中推断出客户端提交的更新向量是两个可能向量中的哪一个。通过进行隐私审核，我们评估了该攻击的成功概率，并量化了SecAgg提供的LDP保证。我们的数字结果揭示了，与普遍声明相反，SecAgg并没有提供私密性。

    arXiv:2403.17775v1 Announce Type: new  Abstract: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offer
    
[^2]: 消费者物联网流量的调查：安全与隐私

    A Survey on Consumer IoT Traffic: Security and Privacy

    [https://arxiv.org/abs/2403.16149](https://arxiv.org/abs/2403.16149)

    本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。

    

    在过去几年里，消费者物联网（CIoT）已经进入了公众生活。尽管CIoT提高了人们日常生活的便利性，但也带来了新的安全和隐私问题。我们尝试通过流量分析这一安全领域中的流行方法，找出研究人员可以从流量分析中了解CIoT安全和隐私方面的内容。本调查从安全和隐私角度探讨了CIoT流量分析中的新特征、CIoT流量分析的最新进展以及尚未解决的挑战。我们从2018年1月至2023年12月收集了310篇与CIoT流量分析有关的安全和隐私角度的论文，总结了识别了CIoT新特征的CIoT流量分析过程。然后，我们根据五个应用目标详细介绍了现有的研究工作：设备指纹识别、用户活动推断、恶意行为检测、隐私泄露以及通信模式识别。

    arXiv:2403.16149v1 Announce Type: cross  Abstract: For the past few years, the Consumer Internet of Things (CIoT) has entered public lives. While CIoT has improved the convenience of people's daily lives, it has also brought new security and privacy concerns. In this survey, we try to figure out what researchers can learn about the security and privacy of CIoT by traffic analysis, a popular method in the security community. From the security and privacy perspective, this survey seeks out the new characteristics in CIoT traffic analysis, the state-of-the-art progress in CIoT traffic analysis, and the challenges yet to be solved. We collected 310 papers from January 2018 to December 2023 related to CIoT traffic analysis from the security and privacy perspective and summarized the process of CIoT traffic analysis in which the new characteristics of CIoT are identified. Then, we detail existing works based on five application goals: device fingerprinting, user activity inference, malicious
    
[^3]: Pandora's White-Box：开放LLMs中训练数据泄漏的增加

    Pandora's White-Box: Increased Training Data Leakage in Open LLMs

    [https://arxiv.org/abs/2402.17012](https://arxiv.org/abs/2402.17012)

    本文对开源大型语言模型（LLMs）进行了隐私攻击研究，提出了首个能同时实现高真正率和低误分类率的预训练LLMs会员推理攻击（MIAs），以及展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。

    

    在本文中，我们对开源的大型语言模型（LLMs）遭受的隐私攻击进行了系统研究，其中对手可以访问模型权重、梯度或损失，试图利用它们来了解底层训练数据。我们的主要结果是针对预训练LLMs的第一个会员推理攻击（MIAs），能够同时实现高TPR和低FPR，并展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。我们考虑了对底层模型的不同访问程度、语言模型的定制化以及攻击者可以使用的资源。在预训练设置中，我们提出了三种新的白盒MIAs：基于梯度范数的攻击、监督神经网络分类器和单步损失比攻击。所有这些都优于现有的黑盒基线，并且我们的.....

    arXiv:2402.17012v1 Announce Type: cross  Abstract: In this paper we undertake a systematic study of privacy attacks against open source Large Language Models (LLMs), where an adversary has access to either the model weights, gradients, or losses, and tries to exploit them to learn something about the underlying training data. Our headline results are the first membership inference attacks (MIAs) against pre-trained LLMs that are able to simultaneously achieve high TPRs and low FPRs, and a pipeline showing that over $50\%$ (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, customization of the language model, and resources available to the attacker. In the pre-trained setting, we propose three new white-box MIAs: an attack based on the gradient norm, a supervised neural network classifier, and a single step loss ratio attack. All outperform existing black-box baselines, and our supervi
    

