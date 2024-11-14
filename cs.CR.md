# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PAPER-HILT: Personalized and Adaptive Privacy-Aware Early-Exit for Reinforcement Learning in Human-in-the-Loop Systems](https://arxiv.org/abs/2403.05864) | PAPER-HILT是针对人机协同系统中隐私保护的创新自适应强化学习策略，通过提前退出方法动态调整隐私保护和系统效用，以适应个体行为模式和偏好。 |
| [^2] | [Attacking LLM Watermarks by Exploiting Their Strengths](https://arxiv.org/abs/2402.16187) | 现有的LLM水印系统虽然具有质量保留、鲁棒性和公开检测API等优点，但也因此容易受到各种攻击，研究者提出了一套实用指南以缓解这些攻击。 |
| [^3] | [ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems.](http://arxiv.org/abs/2201.02775) | 本文研究了垂直联邦学习系统中的对抗性主导输入（ADIs），并证明了其在典型VFL系统中的存在。该研究为防止ADIs的使用提供了方法。 |

# 详细

[^1]: PAPER-HILT：个性化和自适应隐私感知的强化学习提前退出在人机协同系统中的应用

    PAPER-HILT: Personalized and Adaptive Privacy-Aware Early-Exit for Reinforcement Learning in Human-in-the-Loop Systems

    [https://arxiv.org/abs/2403.05864](https://arxiv.org/abs/2403.05864)

    PAPER-HILT是针对人机协同系统中隐私保护的创新自适应强化学习策略，通过提前退出方法动态调整隐私保护和系统效用，以适应个体行为模式和偏好。

    

    强化学习（RL）日益成为人机协同（HITL）应用中的首选方法，因其适应于人类交互的动态特性。然而，在这种环境中整合RL会带来重大的隐私问题，可能会不经意地暴露敏感用户信息。为解决这一问题，我们的论文专注于开发PAPER-HILT，一种创新的自适应RL策略，通过利用专为HITL环境中隐私保护设计的提前退出方法。该方法动态调整隐私保护和系统效用之间的权衡，使其操作适应个人行为模式和偏好。我们主要强调面临处理人类行为的可变和不断发展的挑战，使得静态隐私模型失效。通过其应用，评估了PAPER-HILT的有效性。

    arXiv:2403.05864v1 Announce Type: new  Abstract: Reinforcement Learning (RL) has increasingly become a preferred method over traditional rule-based systems in diverse human-in-the-loop (HITL) applications due to its adaptability to the dynamic nature of human interactions. However, integrating RL in such settings raises significant privacy concerns, as it might inadvertently expose sensitive user information. Addressing this, our paper focuses on developing PAPER-HILT, an innovative, adaptive RL strategy through exploiting an early-exit approach designed explicitly for privacy preservation in HITL environments. This approach dynamically adjusts the tradeoff between privacy protection and system utility, tailoring its operation to individual behavioral patterns and preferences. We mainly highlight the challenge of dealing with the variable and evolving nature of human behavior, which renders static privacy models ineffective. PAPER-HILT's effectiveness is evaluated through its applicati
    
[^2]: 利用其优势攻击LLM水印

    Attacking LLM Watermarks by Exploiting Their Strengths

    [https://arxiv.org/abs/2402.16187](https://arxiv.org/abs/2402.16187)

    现有的LLM水印系统虽然具有质量保留、鲁棒性和公开检测API等优点，但也因此容易受到各种攻击，研究者提出了一套实用指南以缓解这些攻击。

    

    生成模型的进展使得人工智能生成的文本、代码和图片能够在许多应用中模仿人类生成的内容。水印技术旨在将信息嵌入模型的输出中以验证其来源，对于减少对这些人工智能生成内容的滥用非常有用。然而，现有的水印方案仍然令人意外地容易受到攻击。具体而言，我们展示了现有的LLM水印系统共享的可取特性，例如质量保留、鲁棒性和公开检测API，反过来却使这些系统容易遭受各种攻击。我们在常见水印设计选择方面严格研究潜在攻击，并提出了缓解攻击的最佳实践和防御措施——建立了一套嵌入和检测LLM水印的实用指南。

    arXiv:2402.16187v1 Announce Type: cross  Abstract: Advances in generative models have made it possible for AI-generated text, code, and images to mirror human-generated content in many applications. Watermarking, a technique that aims to embed information in the output of a model to verify its source, is useful for mitigating misuse of such AI-generated content. However, existing watermarking schemes remain surprisingly susceptible to attack. In particular, we show that desirable properties shared by existing LLM watermarking systems such as quality preservation, robustness, and public detection APIs can in turn make these systems vulnerable to various attacks. We rigorously study potential attacks in terms of common watermark design choices, and propose best practices and defenses for mitigation -- establishing a set of practical guidelines for embedding and detection of LLM watermarks.
    
[^3]: ADI: 在垂直联邦学习系统中的对抗性主导输入

    ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems. (arXiv:2201.02775v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2201.02775](http://arxiv.org/abs/2201.02775)

    本文研究了垂直联邦学习系统中的对抗性主导输入（ADIs），并证明了其在典型VFL系统中的存在。该研究为防止ADIs的使用提供了方法。

    

    最近，垂直联邦学习（VFL）系统作为处理分散在许多个体来源中的数据的概念而变得突出，无需将其集中化。多个参与者以隐私意识的方式协作训练基于其本地数据的模型。到目前为止，VFL已成为在组织之间安全学习模型的事实解决方案，允许共享知识而不影响任何个人的隐私。尽管VFL系统的发展昌盛，但我们发现某些参与者的输入，称为对抗性主导输入（ADIs），可以支配共同推断朝着对手的意愿方向并迫使其他（受害者）参与者做出微不足道的贡献，失去通常在联邦学习场景中提供的对其贡献重要性的奖励。我们对ADIs进行了系统研究，首先证明了它们在典型的VFL系统中的存在。然后，我们提出了基于梯度的方法

    Vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-aware manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individuals. Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in federated learning scenarios. We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods
    

