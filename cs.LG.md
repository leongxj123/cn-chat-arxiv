# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PAPER-HILT: Personalized and Adaptive Privacy-Aware Early-Exit for Reinforcement Learning in Human-in-the-Loop Systems](https://arxiv.org/abs/2403.05864) | PAPER-HILT是针对人机协同系统中隐私保护的创新自适应强化学习策略，通过提前退出方法动态调整隐私保护和系统效用，以适应个体行为模式和偏好。 |
| [^2] | [Protect Your Score: Contact Tracing With Differential Privacy Guarantees](https://arxiv.org/abs/2312.11581) | 这篇论文提出了具有差分隐私保障的接触追踪算法，以解决隐私问题限制接触追踪的部署。该算法在多种情景下展现了卓越性能，并通过在发布每个风险分数时保护个体健康状况的隐私。 |
| [^3] | [Distributionally Robust Transfer Learning.](http://arxiv.org/abs/2309.06534) | 这篇论文介绍了一种分布鲁棒的迁移学习方法，通过优化一个不确定性集合内最具对抗性的损失来实现，该集合是由源分布的凸组合生成的目标人口集合，能够有效地将迁移学习和分布鲁棒的预测模型联系起来。 |

# 详细

[^1]: PAPER-HILT：个性化和自适应隐私感知的强化学习提前退出在人机协同系统中的应用

    PAPER-HILT: Personalized and Adaptive Privacy-Aware Early-Exit for Reinforcement Learning in Human-in-the-Loop Systems

    [https://arxiv.org/abs/2403.05864](https://arxiv.org/abs/2403.05864)

    PAPER-HILT是针对人机协同系统中隐私保护的创新自适应强化学习策略，通过提前退出方法动态调整隐私保护和系统效用，以适应个体行为模式和偏好。

    

    强化学习（RL）日益成为人机协同（HITL）应用中的首选方法，因其适应于人类交互的动态特性。然而，在这种环境中整合RL会带来重大的隐私问题，可能会不经意地暴露敏感用户信息。为解决这一问题，我们的论文专注于开发PAPER-HILT，一种创新的自适应RL策略，通过利用专为HITL环境中隐私保护设计的提前退出方法。该方法动态调整隐私保护和系统效用之间的权衡，使其操作适应个人行为模式和偏好。我们主要强调面临处理人类行为的可变和不断发展的挑战，使得静态隐私模型失效。通过其应用，评估了PAPER-HILT的有效性。

    arXiv:2403.05864v1 Announce Type: new  Abstract: Reinforcement Learning (RL) has increasingly become a preferred method over traditional rule-based systems in diverse human-in-the-loop (HITL) applications due to its adaptability to the dynamic nature of human interactions. However, integrating RL in such settings raises significant privacy concerns, as it might inadvertently expose sensitive user information. Addressing this, our paper focuses on developing PAPER-HILT, an innovative, adaptive RL strategy through exploiting an early-exit approach designed explicitly for privacy preservation in HITL environments. This approach dynamically adjusts the tradeoff between privacy protection and system utility, tailoring its operation to individual behavioral patterns and preferences. We mainly highlight the challenge of dealing with the variable and evolving nature of human behavior, which renders static privacy models ineffective. PAPER-HILT's effectiveness is evaluated through its applicati
    
[^2]: 保护您的分数：具有差分隐私保障的接触追踪

    Protect Your Score: Contact Tracing With Differential Privacy Guarantees

    [https://arxiv.org/abs/2312.11581](https://arxiv.org/abs/2312.11581)

    这篇论文提出了具有差分隐私保障的接触追踪算法，以解决隐私问题限制接触追踪的部署。该算法在多种情景下展现了卓越性能，并通过在发布每个风险分数时保护个体健康状况的隐私。

    

    2020年和2021年的流行病对经济和社会产生了巨大的影响，研究表明，接触追踪算法可以在早期遏制病毒方面起到关键作用。尽管在更有效的接触追踪算法方面已经取得了重大进展，但我们认为目前的隐私问题阻碍了其部署。接触追踪算法的本质在于传递一个风险分数的通信。然而，恰恰是将这个分数传递给用户，对手可以利用这个分数来评估个体的私人健康状况。我们确定了一个现实的攻击场景，并针对这种攻击提出了具有差分隐私保障的接触追踪算法。该算法在两个最常用的基于代理的COVID19模拟器上进行了测试，并在各种情景下展现了卓越性能，特别是在逼真的测试场景中，同时发布每个风险分数时。

    arXiv:2312.11581v2 Announce Type: replace-cross  Abstract: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score w
    
[^3]: 分布鲁棒的迁移学习

    Distributionally Robust Transfer Learning. (arXiv:2309.06534v1 [cs.LG])

    [http://arxiv.org/abs/2309.06534](http://arxiv.org/abs/2309.06534)

    这篇论文介绍了一种分布鲁棒的迁移学习方法，通过优化一个不确定性集合内最具对抗性的损失来实现，该集合是由源分布的凸组合生成的目标人口集合，能够有效地将迁移学习和分布鲁棒的预测模型联系起来。

    

    许多现有的迁移学习方法依赖于利用与目标数据相似的源数据的信息。然而，这种方法经常忽视了可能存在于不同但潜在相关的辅助样本中的有价值的知识。当处理有限的目标数据和多样化的源模型时，我们的论文引入了一种新颖的方法，分布鲁棒迁移学习（TransDRO），它摆脱了严格的相似性约束。TransDRO通过在一个不确定性集合内优化最具对抗性的损失来设计，该集合定义为由源分布的凸组合生成的目标人口的集合，保证了对目标数据的出色预测性能。TransDRO有效地将迁移学习和分布鲁棒的预测模型联系起来。我们建立了TransDRO的可辨识性和其作为最接近源模型的加权平均值的解释。

    Many existing transfer learning methods rely on leveraging information from source data that closely resembles the target data. However, this approach often overlooks valuable knowledge that may be present in different yet potentially related auxiliary samples. When dealing with a limited amount of target data and a diverse range of source models, our paper introduces a novel approach, Distributionally Robust Optimization for Transfer Learning (TransDRO), that breaks free from strict similarity constraints. TransDRO is designed to optimize the most adversarial loss within an uncertainty set, defined as a collection of target populations generated as a convex combination of source distributions that guarantee excellent prediction performances for the target data. TransDRO effectively bridges the realms of transfer learning and distributional robustness prediction models. We establish the identifiability of TransDRO and its interpretation as a weighted average of source models closest to
    

