# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Consumer IoT Traffic: Security and Privacy](https://arxiv.org/abs/2403.16149) | 本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。 |
| [^2] | [UOR: Universal Backdoor Attacks on Pre-trained Language Models.](http://arxiv.org/abs/2305.09574) | 本文介绍了一种新的后门攻击方法UOR，可以自动选择触发器并学习通用输出表示，成功率高达99.3％，能够对多种预训练语言模型和下游任务实施攻击，且可突破最新的防御方法。 |

# 详细

[^1]: 消费者物联网流量的调查：安全与隐私

    A Survey on Consumer IoT Traffic: Security and Privacy

    [https://arxiv.org/abs/2403.16149](https://arxiv.org/abs/2403.16149)

    本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。

    

    在过去几年里，消费者物联网（CIoT）已经进入了公众生活。尽管CIoT提高了人们日常生活的便利性，但也带来了新的安全和隐私问题。我们尝试通过流量分析这一安全领域中的流行方法，找出研究人员可以从流量分析中了解CIoT安全和隐私方面的内容。本调查从安全和隐私角度探讨了CIoT流量分析中的新特征、CIoT流量分析的最新进展以及尚未解决的挑战。我们从2018年1月至2023年12月收集了310篇与CIoT流量分析有关的安全和隐私角度的论文，总结了识别了CIoT新特征的CIoT流量分析过程。然后，我们根据五个应用目标详细介绍了现有的研究工作：设备指纹识别、用户活动推断、恶意行为检测、隐私泄露以及通信模式识别。

    arXiv:2403.16149v1 Announce Type: cross  Abstract: For the past few years, the Consumer Internet of Things (CIoT) has entered public lives. While CIoT has improved the convenience of people's daily lives, it has also brought new security and privacy concerns. In this survey, we try to figure out what researchers can learn about the security and privacy of CIoT by traffic analysis, a popular method in the security community. From the security and privacy perspective, this survey seeks out the new characteristics in CIoT traffic analysis, the state-of-the-art progress in CIoT traffic analysis, and the challenges yet to be solved. We collected 310 papers from January 2018 to December 2023 related to CIoT traffic analysis from the security and privacy perspective and summarized the process of CIoT traffic analysis in which the new characteristics of CIoT are identified. Then, we detail existing works based on five application goals: device fingerprinting, user activity inference, malicious
    
[^2]: UOR：预训练语言模型的通用后门攻击

    UOR: Universal Backdoor Attacks on Pre-trained Language Models. (arXiv:2305.09574v1 [cs.CL])

    [http://arxiv.org/abs/2305.09574](http://arxiv.org/abs/2305.09574)

    本文介绍了一种新的后门攻击方法UOR，可以自动选择触发器并学习通用输出表示，成功率高达99.3％，能够对多种预训练语言模型和下游任务实施攻击，且可突破最新的防御方法。

    

    在预训练语言模型中植入后门可以传递到各种下游任务，这对安全构成了严重威胁。然而，现有的针对预训练语言模型的后门攻击大都是非目标和特定任务的。很少有针对目标和任务不可知性的方法使用手动预定义的触发器和输出表示，这使得攻击效果不够强大和普适。本文首先总结了一个更具威胁性的预训练语言模型后门攻击应满足的要求，然后提出了一种新的后门攻击方法UOR，通过将手动选择变成自动优化，打破了以往方法的瓶颈。具体来说，我们定义了被污染的监督对比学习，可以自动学习各种预训练语言模型触发器的更加均匀和通用输出表示。此外，我们使用梯度搜索选取适当的触发词，可以适应不同的预训练语言模型和词汇表。实验证明，UOR可以在各种PLMs和下游任务中实现高后门成功率（高达99.3％），优于现有方法。此外，UOR还可以突破对抗后门攻击的最新防御方法。

    Backdoors implanted in pre-trained language models (PLMs) can be transferred to various downstream tasks, which exposes a severe security threat. However, most existing backdoor attacks against PLMs are un-targeted and task-specific. Few targeted and task-agnostic methods use manually pre-defined triggers and output representations, which prevent the attacks from being more effective and general. In this paper, we first summarize the requirements that a more threatening backdoor attack against PLMs should satisfy, and then propose a new backdoor attack method called UOR, which breaks the bottleneck of the previous approach by turning manual selection into automatic optimization. Specifically, we define poisoned supervised contrastive learning which can automatically learn the more uniform and universal output representations of triggers for various PLMs. Moreover, we use gradient search to select appropriate trigger words which can be adaptive to different PLMs and vocabularies. Experi
    

