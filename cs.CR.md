# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring Safety Generalization Challenges of Large Language Models via Code](https://arxiv.org/abs/2403.07865) | 本论文引入了CodeAttack框架用于测试大型语言模型的安全泛化，研究发现GPT-4、Claude-2和Llama-2系列等最新模型存在代码输入的安全漏洞。 |
| [^2] | [Zero-shot sampling of adversarial entities in biomedical question answering](https://arxiv.org/abs/2402.10527) | 在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。 |

# 详细

[^1]: 通过代码探索大型语言模型的安全泛化挑战

    Exploring Safety Generalization Challenges of Large Language Models via Code

    [https://arxiv.org/abs/2403.07865](https://arxiv.org/abs/2403.07865)

    本论文引入了CodeAttack框架用于测试大型语言模型的安全泛化，研究发现GPT-4、Claude-2和Llama-2系列等最新模型存在代码输入的安全漏洞。

    

    大型语言模型（LLMs）的快速发展带来了自然语言处理方面的显著能力，但也引发了人们对它们潜在误用的担忧。本文引入了CodeAttack，一个将自然语言输入转换为代码输入的框架，为测试LLMs的安全泛化提供了一个新颖的环境。我们对包括GPT-4、Claude-2和Llama-2系列在内的最新LLMs进行了全面研究，发现这些模型对于代码输入存在共同的安全漏洞：CodeAttack在超过80%的时间内始终绕过所有模型的安全保护。

    arXiv:2403.07865v1 Announce Type: cross  Abstract: The rapid advancement of Large Language Models (LLMs) has brought about remarkable capabilities in natural language processing but also raised concerns about their potential misuse. While strategies like supervised fine-tuning and reinforcement learning from human feedback have enhanced their safety, these methods primarily focus on natural languages, which may not generalize to other domains. This paper introduces CodeAttack, a framework that transforms natural language inputs into code inputs, presenting a novel environment for testing the safety generalization of LLMs. Our comprehensive studies on state-of-the-art LLMs including GPT-4, Claude-2, and Llama-2 series reveal a common safety vulnerability of these models against code input: CodeAttack consistently bypasses the safety guardrails of all models more than 80\% of the time. Furthermore, we find that a larger distribution gap between CodeAttack and natural language leads to we
    
[^2]: 生物医学问题回答中的零样本采样对抗实体

    Zero-shot sampling of adversarial entities in biomedical question answering

    [https://arxiv.org/abs/2402.10527](https://arxiv.org/abs/2402.10527)

    在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。

    

    大型语言模型（LLM）中参数域知识的增加深度推动它们在现实世界应用中的快速部署。在高风险和知识密集型任务中，理解模型的漏洞对于量化模型预测的可信度和规范其使用至关重要。最近发现在自然语言处理任务中作为对抗示例的命名实体引发了关于它们在其他环境中可能的伪装的疑问。在这里，我们提出了一种在嵌入空间中的幂缩放距离加权采样方案，以发现多样化的对抗实体作为干扰因素。我们展示了它在生物医学主题的对抗性问题回答中优于随机采样的优势。我们的方法使得可以探索攻击表面上的不同区域，这揭示了两种在特征上明显不同的对抗性实体的制度。此外，我们展示了攻击方式如何...

    arXiv:2402.10527v1 Announce Type: new  Abstract: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks su
    

