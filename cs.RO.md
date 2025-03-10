# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Multi-Agent Mapping for Planetary Exploration](https://arxiv.org/abs/2404.02289) | 联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。 |
| [^2] | [On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities](https://arxiv.org/abs/2402.10340) | 论文突出探讨了在机器人应用中整合大型语言模型和视觉语言模型所带来的安全性和健壮性关键问题，指出这种整合可能容易受到恶意攻击并导致严重后果。 |

# 详细

[^1]: 行星探测的联邦多智能体建图

    Federated Multi-Agent Mapping for Planetary Exploration

    [https://arxiv.org/abs/2404.02289](https://arxiv.org/abs/2404.02289)

    联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。

    

    在多智能体机器人探测中，管理和有效利用动态环境产生的大量异构数据构成了一个重要挑战。联邦学习（FL）是一种有前途的分布式映射方法，它解决了协作学习中去中心化数据的挑战。FL使多个智能体之间可以进行联合模型训练，而无需集中化或共享原始数据，克服了带宽和存储限制。我们的方法利用隐式神经映射，将地图表示为由神经网络学习的连续函数，以便实现紧凑和适应性的表示。我们进一步通过在地球数据集上进行元初始化来增强这一方法，预训练网络以快速学习新的地图结构。这种组合在诸如火星地形和冰川等不同领域展现了较强的泛化能力。我们对这一方法进行了严格评估，展示了其有效性。

    arXiv:2404.02289v1 Announce Type: cross  Abstract: In multi-agent robotic exploration, managing and effectively utilizing the vast, heterogeneous data generated from dynamic environments poses a significant challenge. Federated learning (FL) is a promising approach for distributed mapping, addressing the challenges of decentralized data in collaborative learning. FL enables joint model training across multiple agents without requiring the centralization or sharing of raw data, overcoming bandwidth and storage constraints. Our approach leverages implicit neural mapping, representing maps as continuous functions learned by neural networks, for compact and adaptable representations. We further enhance this approach with meta-initialization on Earth datasets, pre-training the network to quickly learn new map structures. This combination demonstrates strong generalization to diverse domains like Martian terrain and glaciers. We rigorously evaluate this approach, demonstrating its effectiven
    
[^2]: 论部署LLMs/VLMs在机器人领域存在的安全问题：突显风险和漏洞

    On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities

    [https://arxiv.org/abs/2402.10340](https://arxiv.org/abs/2402.10340)

    论文突出探讨了在机器人应用中整合大型语言模型和视觉语言模型所带来的安全性和健壮性关键问题，指出这种整合可能容易受到恶意攻击并导致严重后果。

    

    在这篇论文中，我们着重讨论了将大型语言模型（LLMs）和视觉语言模型（VLMs）整合到机器人应用中所涉及的健壮性和安全性关键问题。最近的研究着重于利用LLMs和VLMs来提高机器人任务（如操作，导航等）的性能。然而，这种整合可能会引入显着的漏洞，即由于语言模型对恶意攻击的敏感性，可能导致灾难性后果。通过研究LLMs/VLMs与机器人界面的最新进展，我们展示了如何轻松操纵或误导机器人的行为，导致安全隐患。我们定义并提供了几种可能的恶意攻击示例，并对集成了语言模型的三个知名机器人框架（包括KnowNo VIMA和Instruct2Act）进行实验，以评估它们对这些攻击的敏感度。

    arXiv:2402.10340v1 Announce Type: cross  Abstract: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these atta
    

