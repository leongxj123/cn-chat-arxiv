# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Words to Routes: Applying Large Language Models to Vehicle Routing](https://arxiv.org/abs/2403.10795) | 该研究探索了应用大型语言模型解决车辆路径规划问题的能力，提出了基于自然语言任务描述的基本提示范例并提出一种使模型进行改进的框架。 |

# 详细

[^1]: 从单词到路径：将大型语言模型应用于车辆路径规划

    From Words to Routes: Applying Large Language Models to Vehicle Routing

    [https://arxiv.org/abs/2403.10795](https://arxiv.org/abs/2403.10795)

    该研究探索了应用大型语言模型解决车辆路径规划问题的能力，提出了基于自然语言任务描述的基本提示范例并提出一种使模型进行改进的框架。

    

    LLMs在机器人领域（例如操作和导航）中展示出令人印象深刻的进展，其利用自然语言任务描述。LLMs在这些任务中取得成功让我们思考：LLMs在使用自然语言任务描述解决车辆路径规划问题（VRPs）的能力如何？在这项工作中，我们分三步研究这个问题。首先，我们构建了一个包含21种单车或多车路径问题的数据集。其次，我们评估了LLMs在四种基本提示范例文本到代码生成的性能，每种包括不同类型的文本输入。我们发现，直接从自然语言任务描述生成代码的基本提示范例对于GPT-4效果最佳，实现了56%的可行性，40%的优化性和53%的效率。第三，基于观察到LLMs可能无法在初始尝试中提供正确解决方案的现象，我们提出了一个框架，使LLMs能够进行改进。

    arXiv:2403.10795v1 Announce Type: cross  Abstract: LLMs have shown impressive progress in robotics (e.g., manipulation and navigation) with natural language task descriptions. The success of LLMs in these tasks leads us to wonder: What is the ability of LLMs to solve vehicle routing problems (VRPs) with natural language task descriptions? In this work, we study this question in three steps. First, we construct a dataset with 21 types of single- or multi-vehicle routing problems. Second, we evaluate the performance of LLMs across four basic prompt paradigms of text-to-code generation, each involving different types of text input. We find that the basic prompt paradigm, which generates code directly from natural language task descriptions, performs the best for GPT-4, achieving 56% feasibility, 40% optimality, and 53% efficiency. Third, based on the observation that LLMs may not be able to provide correct solutions at the initial attempt, we propose a framework that enables LLMs to refin
    

