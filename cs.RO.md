# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Temporal and Semantic Evaluation Metrics for Foundation Models in Post-Hoc Analysis of Robotic Sub-tasks](https://arxiv.org/abs/2403.17238) | 提出了一种基于基础模型的自动化框架，通过新颖的提示策略将轨迹数据分解为时间和语言描述的子任务，同时引入了时间相似性和语义相似性两种新的评估指标。 |

# 详细

[^1]: 基于时间和语义评估指标的基础模型在机器人子任务事后分析中的应用

    Temporal and Semantic Evaluation Metrics for Foundation Models in Post-Hoc Analysis of Robotic Sub-tasks

    [https://arxiv.org/abs/2403.17238](https://arxiv.org/abs/2403.17238)

    提出了一种基于基础模型的自动化框架，通过新颖的提示策略将轨迹数据分解为时间和语言描述的子任务，同时引入了时间相似性和语义相似性两种新的评估指标。

    

    最近在任务和运动规划（TAMP）领域的研究表明，在使用带有质量标记数据的语言监督机器人轨迹进行控制策略训练可以显着提高代理任务成功率。然而，这类数据的稀缺性对将这些方法扩展到一般用例构成重大障碍。为了解决这一问题，我们提出了一种自动化框架，通过利用最近的基础模型（FMs）的提示策略，包括大型语言模型（LLMs）和视觉语言模型（VLMs），将轨迹数据分解为基于时间和自然语言的描述性子任务。我们的框架为构成完整轨迹的底层子任务提供了基于时间和语言的描述。为了严格评估我们的自动标记框架的质量，我们提出了一种算法 SIMILARITY 来生成两种新颖的指标，即时间相似性和语义相似性。

    arXiv:2403.17238v1 Announce Type: cross  Abstract: Recent works in Task and Motion Planning (TAMP) show that training control policies on language-supervised robot trajectories with quality labeled data markedly improves agent task success rates. However, the scarcity of such data presents a significant hurdle to extending these methods to general use cases. To address this concern, we present an automated framework to decompose trajectory data into temporally bounded and natural language-based descriptive sub-tasks by leveraging recent prompting strategies for Foundation Models (FMs) including both Large Language Models (LLMs) and Vision Language Models (VLMs). Our framework provides both time-based and language-based descriptions for lower-level sub-tasks that comprise full trajectories. To rigorously evaluate the quality of our automatic labeling framework, we contribute an algorithm SIMILARITY to produce two novel metrics, temporal similarity and semantic similarity. The metrics me
    

