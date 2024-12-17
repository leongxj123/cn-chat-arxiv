# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning](https://arxiv.org/abs/2404.00776) | PyTorch Frame是一个用于处理多模态表格数据的PyTorch框架，通过提供数据结构、模型抽象和外部基础模型整合等功能，实现了模块化的表格模型实现，并成功将这些模型应用于复杂的数据集。 |
| [^2] | [HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding](https://arxiv.org/abs/2403.19723) | HGT框架结合了异质图增强的大型语言模型，通过软提示和多粒度自监督HG预训练目标，实现了少样本复杂表格理解任务的最新成果。 |

# 详细

[^1]: PyTorch Frame: 一个用于多模态表格学习的模块化框架

    PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning

    [https://arxiv.org/abs/2404.00776](https://arxiv.org/abs/2404.00776)

    PyTorch Frame是一个用于处理多模态表格数据的PyTorch框架，通过提供数据结构、模型抽象和外部基础模型整合等功能，实现了模块化的表格模型实现，并成功将这些模型应用于复杂的数据集。

    

    我们提出了PyTorch Frame，这是一个基于PyTorch的框架，用于处理多模态表格数据的深度学习。PyTorch Frame通过提供基于PyTorch的数据结构来处理复杂的表格数据，引入模型抽象以实现表格模型的模块化实现，并允许整合外部基础模型来处理复杂列（例如，用于文本列的LLMs）。我们通过以模块化方式实现多样的表格模型，成功将这些模型应用于复杂的多模态表格数据，并将我们的框架与PyTorch Geometric集成，PyTorch Geometric是一个用于图神经网络（GNNs）的PyTorch库，以实现对关系数据库的端到端学习。

    arXiv:2404.00776v1 Announce Type: new  Abstract: We present PyTorch Frame, a PyTorch-based framework for deep learning over multi-modal tabular data. PyTorch Frame makes tabular deep learning easy by providing a PyTorch-based data structure to handle complex tabular data, introducing a model abstraction to enable modular implementation of tabular models, and allowing external foundation models to be incorporated to handle complex columns (e.g., LLMs for text columns). We demonstrate the usefulness of PyTorch Frame by implementing diverse tabular models in a modular way, successfully applying these models to complex multi-modal tabular data, and integrating our framework with PyTorch Geometric, a PyTorch library for Graph Neural Networks (GNNs), to perform end-to-end learning over relational databases.
    
[^2]: HGT：利用异质图增强的大型语言模型进行少样本复杂表格理解

    HGT: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding

    [https://arxiv.org/abs/2403.19723](https://arxiv.org/abs/2403.19723)

    HGT框架结合了异质图增强的大型语言模型，通过软提示和多粒度自监督HG预训练目标，实现了少样本复杂表格理解任务的最新成果。

    

    表格理解 (TU) 取得了显著进展，但面临手动标记表格的稀缺性和复杂表格结构的挑战。为解决这些问题，我们提出了 HGT 框架，其中包含一个异质图 (HG) 增强的大型语言模型 (LLM)，用于解决少样本 TU 任务。它通过软提示和指导转换将表格语义与LLM的参数化知识对齐，并通过涉及三种新的多粒度自监督HG预训练目标的多任务预训练方案处理复杂表格。我们在几个基准测试上通过实证方法展示了HGT的有效性，表明它在少样本复杂TU方面的表现优于SOTA。

    arXiv:2403.19723v1 Announce Type: cross  Abstract: Table understanding (TU) has achieved promising advancements, but it faces the challenges of the scarcity of manually labeled tables and the presence of complex table structures.To address these challenges, we propose HGT, a framework with a heterogeneous graph (HG)-enhanced large language model (LLM) to tackle few-shot TU tasks.It leverages the LLM by aligning the table semantics with the LLM's parametric knowledge through soft prompts and instruction turning and deals with complex tables by a multi-task pre-training scheme involving three novel multi-granularity self-supervised HG pre-training objectives.We empirically demonstrate the effectiveness of HGT, showing that it outperforms the SOTA for few-shot complex TU on several benchmarks.
    

