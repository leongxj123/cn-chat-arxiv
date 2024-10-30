# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pre-training Differentially Private Models with Limited Public Data](https://arxiv.org/abs/2402.18752) | 通过使用有限的公共数据，研究者提出了一种新颖的差分隐私持续预训练策略，以显著缓解优化器性能下降。 |
| [^2] | [Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents](https://arxiv.org/abs/2402.11208) | 这项工作调查了基于LLM的代理人面临的后门攻击威胁，并提出了一般框架和不同形式的后门攻击分析。 |

# 详细

[^1]: 使用有限公共数据对有差异隐私的模型进行预训练

    Pre-training Differentially Private Models with Limited Public Data

    [https://arxiv.org/abs/2402.18752](https://arxiv.org/abs/2402.18752)

    通过使用有限的公共数据，研究者提出了一种新颖的差分隐私持续预训练策略，以显著缓解优化器性能下降。

    

    大型基础模型卓越的性能依赖于大量高质量数据的使用，然而这些数据通常包含需要正式保护的敏感、私人和受版权保护的信息。差分隐私是一种用于衡量模型提供的安全程度的重要方法，然而由于在预训练阶段应用差分隐私会导致性能下降，因此其应用通常仅限于模型微调阶段。因此，差分隐私目前尚不能保护初始预训练过程中使用的大部分数据。在这项工作中，我们首先通过分析每次迭代的损失改进，对差分隐私训练的有效性提供了理论理解。我们观察到，通过使用有限的公共数据，可以显著缓解差分隐私优化器的性能下降，从而引出一种新颖的差分隐私持续预训练策略。在实证方面，通过

    arXiv:2402.18752v1 Announce Type: new  Abstract: The superior performance of large foundation models relies on the use of massive amounts of high-quality data, which often contain sensitive, private and copyrighted material that requires formal protection. While differential privacy (DP) is a prominent method to gauge the degree of security provided to the models, its application is commonly limited to the model fine-tuning stage, due to the performance degradation when applying DP during the pre-training stage. Consequently, DP is yet not capable of protecting a substantial portion of the data used during the initial pre-training process.   In this work, we first provide a theoretical understanding of the efficacy of DP training by analyzing the per-iteration loss improvement. We make a key observation that DP optimizers' performance degradation can be significantly mitigated by the use of limited public data, which leads to a novel DP continual pre-training strategy. Empirically, usi
    
[^2]: 警惕您的代理人！调查基于LLM的代理人的后门威胁

    Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents

    [https://arxiv.org/abs/2402.11208](https://arxiv.org/abs/2402.11208)

    这项工作调查了基于LLM的代理人面临的后门攻击威胁，并提出了一般框架和不同形式的后门攻击分析。

    

    利用大型语言模型LLM的快速发展，已经开发出了用于处理各种实际应用（包括金融、医疗保健和购物等）的基于LLM的代理人。在应用过程中确保LLM代理人的可靠性和安全性至关重要。然而，目前对LLM代理人的安全性问题尚未得到充分探讨。本工作首次探讨了典型安全威胁之一，即对LLM代理人的后门攻击。我们首先制定了一个代理人后门攻击的一般框架，然后对不同形式的代理人后门攻击进行了彻底分析。具体而言，从最终攻击结果的角度来看，攻击者可以选择操纵最终输出分布，或者仅在中间推理过程中引入恶意行为，同时保持最终输出的正确性。此外，前一类可以分为

    arXiv:2402.11208v1 Announce Type: cross  Abstract: Leveraging the rapid development of Large Language Models LLMs, LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis on the different forms of agent backdoor attacks. Specifically, from the perspective of the final attacking outcomes, the attacker can either choose to manipulate the final output distribution, or only introduce malicious behavior in the intermediate reasoning process, while keeping the final output correct. Furthermore, the former category can be divided
    

