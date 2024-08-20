# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities](https://arxiv.org/abs/2305.13168) | 本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。 |
| [^2] | [UniTS: A Universal Time Series Analysis Framework with Self-supervised Representation Learning.](http://arxiv.org/abs/2303.13804) | UniTS是一个带自监督表示学习的通用时间序列分析框架，能够解决部分标记和领域转移等实际问题，并在多个任务和设置中实现了优秀的性能表现。 |

# 详细

[^1]: LLMs用于知识图谱构建和推理：最新功能与未来机遇

    LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities

    [https://arxiv.org/abs/2305.13168](https://arxiv.org/abs/2305.13168)

    本研究全面评估了LLMs在知识图谱构建和推理领域的性能，发现GPT-4更适合作为推理助手，并在某些情况下超越了精调模型。

    

    本文对大规模语言模型（LLMs）在知识图谱（KG）构建和推理中的数量化和质化评估进行了详尽的研究。我们在八个不同的数据集上进行了实验，重点关注涵盖实体和关系提取、事件提取、链接预测和问答四个典型任务，从而全面探索了LLMs在构建和推理领域的表现。经验性研究发现，以GPT-4为代表的LLMs更适合作为推理助手，而不是少样本信息提取器。具体而言，虽然GPT-4在与KG构建相关的任务中表现出色，但在推理任务中表现更出色，在某些情况下超越了精调模型。此外，我们的调查还扩展到LLMs在信息提取方面的潜在泛化能力，提出了虚拟知识提取的构想。

    arXiv:2305.13168v2 Announce Type: replace-cross  Abstract: This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We engage in experiments across eight diverse datasets, focusing on four representative tasks encompassing entity and relation extraction, event extraction, link prediction, and question-answering, thereby thoroughly exploring LLMs' performance in the domain of construction and inference. Empirically, our findings suggest that LLMs, represented by GPT-4, are more suited as inference assistants rather than few-shot information extractors. Specifically, while GPT-4 exhibits good performance in tasks related to KG construction, it excels further in reasoning tasks, surpassing fine-tuned models in certain cases. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, leading to the proposition of a Virtual Knowledge Extr
    
[^2]: UniTS: 一种带自监督表示学习的通用时间序列分析框架

    UniTS: A Universal Time Series Analysis Framework with Self-supervised Representation Learning. (arXiv:2303.13804v1 [cs.LG])

    [http://arxiv.org/abs/2303.13804](http://arxiv.org/abs/2303.13804)

    UniTS是一个带自监督表示学习的通用时间序列分析框架，能够解决部分标记和领域转移等实际问题，并在多个任务和设置中实现了优秀的性能表现。

    

    机器学习已经成为时间序列分析的强有力工具。现有方法通常针对不同分析任务进行定制，并面临着处理部分标记和领域转移等实际问题的挑战。为了实现通用分析并解决上述问题，我们开发了UniTS，这是一个新颖的框架，它集成了自监督表示学习（或预训练）。 UniTS的组件使用类似于sklearn的API进行设计，以允许灵活的扩展。我们演示了用户如何使用用户友好的GUI执行分析任务，并展示了UniTS在五个主流任务和两个实际设置中相较于传统特定任务方法没有自监督预训练的卓越性能。

    Machine learning has emerged as a powerful tool for time series analysis. Existing methods are usually customized for different analysis tasks and face challenges in tackling practical problems such as partial labeling and domain shift. To achieve universal analysis and address the aforementioned problems, we develop UniTS, a novel framework that incorporates self-supervised representation learning (or pre-training). The components of UniTS are designed using sklearn-like APIs to allow flexible extensions. We demonstrate how users can easily perform an analysis task using the user-friendly GUIs, and show the superior performance of UniTS over the traditional task-specific methods without self-supervised pre-training on five mainstream tasks and two practical settings.
    

