# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OmniPred: Language Models as Universal Regressors](https://arxiv.org/abs/2402.14547) | 本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。 |
| [^2] | [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168) | 本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。 |

# 详细

[^1]: OmniPred：语言模型作为通用回归器

    OmniPred: Language Models as Universal Regressors

    [https://arxiv.org/abs/2402.14547](https://arxiv.org/abs/2402.14547)

    本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。

    

    在实验设计的广阔领域中，回归一直是一个强大的工具，可以准确预测系统或模型在给定一组参数的情况下的结果指标，但传统上只限于适用于特定任务的方法。在本文中，我们提出了OmniPred，这是一个用于训练语言模型作为通用端到端回归器的框架，使用来自多样真实世界实验的$(x,y)$评估数据。通过使用源自Google Vizier的数据，这是世界上最大的黑盒优化数据库之一，我们的大量实验表明，仅通过数学参数和值的文本表示，语言模型能够进行非常精确的数值回归，如果有机会训练多个任务，则可以显著优于传统的回归模型。

    arXiv:2402.14547v1 Announce Type: cross  Abstract: Over the broad landscape of experimental design, regression has been a powerful tool to accurately predict the outcome metrics of a system or model given a set of parameters, but has been traditionally restricted to methods which are only applicable to a specific task. In this paper, we propose OmniPred, a framework for training language models as universal end-to-end regressors over $(x,y)$ evaluation data from diverse real world experiments. Using data sourced from Google Vizier, one of the largest blackbox optimization databases in the world, our extensive experiments demonstrate that through only textual representations of mathematical parameters and values, language models are capable of very precise numerical regression, and if given the opportunity to train over multiple tasks, can significantly outperform traditional regression models.
    
[^2]: LLMs用于知识图谱构建和推理：最新功能与未来机遇

    LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities

    [https://arxiv.org/abs/2305.13168](https://arxiv.org/abs/2305.13168)

    本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。

    

    本文对大规模语言模型（LLMs）在知识图谱（KG）构建和推理中的数量化和质化评估进行了详尽的研究。我们在八个不同的数据集上进行了实验，重点关注涵盖实体和关系提取、事件提取、链接预测和问答四个典型任务，从而全面探索了LLMs在构建和推理领域的表现。经验性研究发现，以GPT-4为代表的LLMs更适合作为推理助手，而不是少样本信息提取器。具体而言，虽然GPT-4在与KG构建相关的任务中表现出色，但在推理任务中表现更出色，在某些情况下超越了精调模型。此外，我们的调查还扩展到LLMs在信息提取方面的潜在泛化能力，提出了虚拟知识提取的构想。

    arXiv:2305.13168v2 Announce Type: replace-cross  Abstract: This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We engage in experiments across eight diverse datasets, focusing on four representative tasks encompassing entity and relation extraction, event extraction, link prediction, and question-answering, thereby thoroughly exploring LLMs' performance in the domain of construction and inference. Empirically, our findings suggest that LLMs, represented by GPT-4, are more suited as inference assistants rather than few-shot information extractors. Specifically, while GPT-4 exhibits good performance in tasks related to KG construction, it excels further in reasoning tasks, surpassing fine-tuned models in certain cases. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, leading to the proposition of a Virtual Knowledge Extr
    

