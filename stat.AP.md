# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-shot sampling of adversarial entities in biomedical question answering](https://arxiv.org/abs/2402.10527) | 在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。 |
| [^2] | [Federated Epidemic Surveillance.](http://arxiv.org/abs/2307.02616) | 本研究旨在探索联邦方法在流行病监测中的应用。我们提出了一个假设检验框架，通过推送到保管人的防火墙并进行元分析，来解决数据分布和共享限制的问题。通过实验验证了我们的方法的有效性，并提出了适合的$p$-值合并方法。这些发现为联邦流行病监测提供了有价值的见解。 |

# 详细

[^1]: 生物医学问题回答中的零样本采样对抗实体

    Zero-shot sampling of adversarial entities in biomedical question answering

    [https://arxiv.org/abs/2402.10527](https://arxiv.org/abs/2402.10527)

    在生物医学问题回答中，我们提出了一种在嵌入空间中进行零样本采样的方案，用于发现各种对抗实体作为干扰因素，相比随机采样，在对抗问答中表现出明显优势，揭示了不同特征的两种对抗性实体制度。

    

    大型语言模型（LLM）中参数域知识的增加深度推动它们在现实世界应用中的快速部署。在高风险和知识密集型任务中，理解模型的漏洞对于量化模型预测的可信度和规范其使用至关重要。最近发现在自然语言处理任务中作为对抗示例的命名实体引发了关于它们在其他环境中可能的伪装的疑问。在这里，我们提出了一种在嵌入空间中的幂缩放距离加权采样方案，以发现多样化的对抗实体作为干扰因素。我们展示了它在生物医学主题的对抗性问题回答中优于随机采样的优势。我们的方法使得可以探索攻击表面上的不同区域，这揭示了两种在特征上明显不同的对抗性实体的制度。此外，我们展示了攻击方式如何...

    arXiv:2402.10527v1 Announce Type: new  Abstract: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks su
    
[^2]: 联邦流行病监测

    Federated Epidemic Surveillance. (arXiv:2307.02616v1 [stat.AP])

    [http://arxiv.org/abs/2307.02616](http://arxiv.org/abs/2307.02616)

    本研究旨在探索联邦方法在流行病监测中的应用。我们提出了一个假设检验框架，通过推送到保管人的防火墙并进行元分析，来解决数据分布和共享限制的问题。通过实验验证了我们的方法的有效性，并提出了适合的$p$-值合并方法。这些发现为联邦流行病监测提供了有价值的见解。

    

    流行病的监测是一项具有挑战性的任务，特别是当关键数据分散且利益相关方无法或不愿共享时。为了克服这一障碍，应开发联邦方法来整合实体愿意提供的较不敏感的证据。本研究旨在探索将假设检验推送到每个保管人的防火墙后，再通过元分析来合并结果的可行性，并确定重建假设检验和优化推理的最佳方法。我们提出了一个假设检验框架来识别指标的激增，并对真实数据和半合成数据进行功效分析和实验，以展示我们所提出的假设检验的性质，并提出合适的$p$-值合并方法。我们的研究结果凸显了使用$p$-值合并作为流行病监测的联邦方法的潜力，并为整合可用信息提供了宝贵的见解。

    The surveillance of a pandemic is a challenging task, especially when crucial data is distributed and stakeholders cannot or are unwilling to share. To overcome this obstacle, federated methodologies should be developed to incorporate less sensitive evidence that entities are willing to provide. This study aims to explore the feasibility of pushing hypothesis tests behind each custodian's firewall and then meta-analysis to combine the results, and to determine the optimal approach for reconstructing the hypothesis test and optimizing the inference. We propose a hypothesis testing framework to identify a surge in the indicators and conduct power analyses and experiments on real and semi-synthetic data to showcase the properties of our proposed hypothesis test and suggest suitable methods for combining $p$-values. Our findings highlight the potential of using $p$-value combination as a federated methodology for pandemic surveillance and provide valuable insights into integrating availabl
    

