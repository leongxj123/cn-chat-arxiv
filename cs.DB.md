# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jellyfish: A Large Language Model for Data Preprocessing](https://arxiv.org/abs/2312.01678) | 这项研究探讨了在数据挖掘中利用大型语言模型进行数据预处理的方法，通过指导调整本地LLMs来解决通用数据预处理问题，确保数据安全并进行进一步调整 |

# 详细

[^1]: Jellyfish：一个用于数据预处理的大型语言模型

    Jellyfish: A Large Language Model for Data Preprocessing

    [https://arxiv.org/abs/2312.01678](https://arxiv.org/abs/2312.01678)

    这项研究探讨了在数据挖掘中利用大型语言模型进行数据预处理的方法，通过指导调整本地LLMs来解决通用数据预处理问题，确保数据安全并进行进一步调整

    

    这篇论文探讨了在数据挖掘管道中将原始数据转换为有利于简单处理的干净格式的数据预处理（DP）中LLMs的利用。与使用LLMs为DP设计通用解决方案引起了兴趣相比，最近在这一领域的倡议通常依赖于GPT API，引发了不可避免的数据泄霏担忧。与这些方法不同，我们考虑将指导调整本地LLMs（7-13B模型）作为通用DP问解器。我们选择了代表性DP任务的四组数据集，并利用针对DP定制的序列化和知识注入技术构建了指导调整数据。因此，指导调整的LLMs使用户能够为DP手动制定指导。同时，它们可以在本地、单一和价格低廉的GPU上运行，确保数据安全并实现进一步调整。我们的实验表明，我们为DP指导构建的数据集

    arXiv:2312.01678v4 Announce Type: replace  Abstract: This paper explores the utilization of LLMs for data preprocessing (DP), a crucial step in the data mining pipeline that transforms raw data into a clean format conducive to easy processing. Whereas the use of LLMs has sparked interest in devising universal solutions to DP, recent initiatives in this domain typically rely on GPT APIs, raising inevitable data breach concerns. Unlike these approaches, we consider instruction-tuning local LLMs (7 - 13B models) as universal DP ask solver. We select a collection of datasets across four representative DP tasks and construct instruction-tuning data using serialization and knowledge injection techniques tailored to DP. As such, the instruction-tuned LLMs empower users to manually craft instructions for DP. Meanwhile, they can operate on a local, single, and low-priced GPU, ensuring data security and enabling further tuning. Our experiments show that our dataset constructed for DP instruction
    

