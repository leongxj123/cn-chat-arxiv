# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks](https://arxiv.org/abs/2403.04783) | 提出了一种基于响应过滤的多Agent防御框架AutoDefense，可以有效提高LLMs对抗越狱攻击的鲁棒性，同时保持正常用户请求的性能。 |

# 详细

[^1]: AutoDefense: 多Agent LLM 防御对抗越狱攻击

    AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks

    [https://arxiv.org/abs/2403.04783](https://arxiv.org/abs/2403.04783)

    提出了一种基于响应过滤的多Agent防御框架AutoDefense，可以有效提高LLMs对抗越狱攻击的鲁棒性，同时保持正常用户请求的性能。

    

    尽管在道德对齐方面进行了广泛的预训练和微调以防止在用户请求时生成有害信息，但大型语言模型（LLMs）仍然容易受到越狱攻击。 本文提出了一种基于响应过滤的多Agent防御框架AutoDefense，用于从LLMs中过滤有害回复。 此框架为LLM代理分配不同角色，并利用它们共同完成防御任务。 任务的划分增强了LLMs的整体遵循指令能力，并使其他防御组件作为工具集成成为可能。 AutoDefense 可以适应各种规模和种类的开源LLMs作为代理。 通过对大量有害和安全提示进行广泛实验，我们验证了所提出的AutoDefense在提高对抗越狱攻击的鲁棒性的同时，保持了正常用户请求的性能。

    arXiv:2403.04783v1 Announce Type: cross  Abstract: Despite extensive pre-training and fine-tuning in moral alignment to prevent generating harmful information at user request, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a response-filtering based multi-agent defense framework that filters harmful responses from LLMs. This framework assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. AutoDefense can adapt to various sizes and kinds of open-source LLMs that serve as agents. Through conducting extensive experiments on a large scale of harmful and safe prompts, we validate the effectiveness of the proposed AutoDefense in improving the robustness against jailbreak attacks, while maintaining the performance at normal user request. Our code and 
    

