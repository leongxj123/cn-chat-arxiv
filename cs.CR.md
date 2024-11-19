# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Subgraph Learning by Monitoring Early Training Representations](https://arxiv.org/abs/2403.09901) | 本文引入了一种名为SHERD的新技术，通过监控图神经网络(GNNs)早期训练表示中的信息，利用标准距离度量检测易受攻击节点，从而在图输入中实现性能和对抗鲁棒性。 |
| [^2] | [Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume](https://arxiv.org/abs/2403.05100) | 提出新指标对抗超体积来全面评估深度学习模型在多种扰动强度下的鲁棒性，并采用新型训练算法来提高对抗鲁棒性。 |
| [^3] | [How (un)ethical are instruction-centric responses of LLMs? Unveiling the vulnerabilities of safety guardrails to harmful queries](https://arxiv.org/abs/2402.15302) | 本研究探讨了大型语言模型（LLMs）对指令中心响应的容忍度，并提出了一个包含复杂查询的数据集，旨在揭示触发不道德响应的方法。 |

# 详细

[^1]: 通过监控早期训练表示来实现鲁棒的子图学习

    Robust Subgraph Learning by Monitoring Early Training Representations

    [https://arxiv.org/abs/2403.09901](https://arxiv.org/abs/2403.09901)

    本文引入了一种名为SHERD的新技术，通过监控图神经网络(GNNs)早期训练表示中的信息，利用标准距离度量检测易受攻击节点，从而在图输入中实现性能和对抗鲁棒性。

    

    引文:2403.09901v1 公告类型:新摘要:图神经网络(GNNs)因在图学习和节点分类任务中表现出色而引起了广泛关注。然而，它们对对抗性攻击的脆弱性，特别是通过易受攻击的节点，给决策制定带来了挑战。鲁棒的图摘要需求在于对抗性挑战会导致攻击在整个图中传播。在本文中，我们通过引入新颖的技术SHERD (通过早期训练表示距离进行子图学习)来解决图输入中的性能和对抗鲁棒性。SHERD利用部分训练的图卷积网络(GCN)的层信息，通过标准距离度量来检测对抗攻击期间易受攻击的节点。该方法识别出"易受攻击的(坏)"节点并移除这些节点，形成一个鲁棒的子图，同时保持节点分类性能。

    arXiv:2403.09901v1 Announce Type: new  Abstract: Graph neural networks (GNNs) have attracted significant attention for their outstanding performance in graph learning and node classification tasks. However, their vulnerability to adversarial attacks, particularly through susceptible nodes, poses a challenge in decision-making. The need for robust graph summarization is evident in adversarial challenges resulting from the propagation of attacks throughout the entire graph. In this paper, we address both performance and adversarial robustness in graph input by introducing the novel technique SHERD (Subgraph Learning Hale through Early Training Representation Distances). SHERD leverages information from layers of a partially trained graph convolutional network (GCN) to detect susceptible nodes during adversarial attacks using standard distance metrics. The method identifies "vulnerable (bad)" nodes and removes such nodes to form a robust subgraph while maintaining node classification perf
    
[^2]: 探索对抗界限：通过对抗超体积量化鲁棒性

    Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume

    [https://arxiv.org/abs/2403.05100](https://arxiv.org/abs/2403.05100)

    提出新指标对抗超体积来全面评估深度学习模型在多种扰动强度下的鲁棒性，并采用新型训练算法来提高对抗鲁棒性。

    

    在深度学习模型面临日益严重的对抗攻击威胁，特别是在安全关键领域，强调了对鲁棒深度学习系统的需求。传统的鲁棒性评估依赖于对抗准确性，该指标衡量模型在特定扰动强度下的性能。然而，这一单一指标并不能完全概括模型对不同程度扰动的整体韧性。为了填补这一空白，我们提出了一种新的指标，称为对抗超体积，从多目标优化的角度综合评估了深度学习模型在一系列扰动强度下的鲁棒性。该指标允许深入比较防御机制，并承认了较弱的防御策略所带来的鲁棒性改进。此外，我们采用了一种提高对抗鲁棒性均匀性的新型训练算法。

    arXiv:2403.05100v1 Announce Type: cross  Abstract: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly
    
[^3]: 有关LLMs指令中心响应的（不道德）程度有多高？揭示安全防护栏对有害查询的漏洞

    How (un)ethical are instruction-centric responses of LLMs? Unveiling the vulnerabilities of safety guardrails to harmful queries

    [https://arxiv.org/abs/2402.15302](https://arxiv.org/abs/2402.15302)

    本研究探讨了大型语言模型（LLMs）对指令中心响应的容忍度，并提出了一个包含复杂查询的数据集，旨在揭示触发不道德响应的方法。

    

    在这项研究中，我们解决了一个围绕大型语言模型（LLMs）安全和道德使用日益关注的问题。尽管这些模型具有潜力，但它们可能会被各种复杂的方法欺骗，产生有害或不道德内容，包括“越狱”技术和有针对性的操纵。我们的工作集中在一个特定问题上：LLMs在要求它们生成以伪代码、程序或软件片段为中心的响应时，有多大程度上可能会被误导，而不是生成普通文本。为了调查这个问题，我们引入了TechHazardQA，一个数据集，其中包含应以文本和以指令为中心格式（例如伪代码）回答的复杂查询，旨在识别不道德响应的触发器。我们查询了一系列LLMs-- Llama-2-13b，Llama-2-7b，Mistral-V2和Mistral 8X7B--并要求它们生成文本和指令为中心的响应。为了评估我们的方法，

    arXiv:2402.15302v1 Announce Type: new  Abstract: In this study, we tackle a growing concern around the safety and ethical use of large language models (LLMs). Despite their potential, these models can be tricked into producing harmful or unethical content through various sophisticated methods, including 'jailbreaking' techniques and targeted manipulation. Our work zeroes in on a specific issue: to what extent LLMs can be led astray by asking them to generate responses that are instruction-centric such as a pseudocode, a program or a software snippet as opposed to vanilla text. To investigate this question, we introduce TechHazardQA, a dataset containing complex queries which should be answered in both text and instruction-centric formats (e.g., pseudocodes), aimed at identifying triggers for unethical responses. We query a series of LLMs -- Llama-2-13b, Llama-2-7b, Mistral-V2 and Mistral 8X7B -- and ask them to generate both text and instruction-centric responses. For evaluation we rep
    

