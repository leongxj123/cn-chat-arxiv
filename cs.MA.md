# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101) | KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。 |

# 详细

[^1]: KnowAgent: 知识增强规划用于基于LLM的Agent

    KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents

    [https://arxiv.org/abs/2403.03101](https://arxiv.org/abs/2403.03101)

    KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。

    

    大型语言模型(LLMs)在复杂推理任务中表现出巨大潜力，但在处理更复杂的挑战时仍有所不足，特别是与环境互动通过生成可执行动作时。这种不足主要来自于语言Agent中缺乏内置动作知识，导致在任务求解过程中无法有效引导规划轨迹，从而导致规划幻觉。为了解决这个问题，我们引入了KnowAgent，一种旨在通过整合显式动作知识来增强LLM规划能力的新方法。具体而言，KnowAgent采用了一个动作知识库和一个知识型自学习策略来限制规划过程中的行动路径，实现更合理的轨迹合成，进而提高语言Agent的计划性能。基于HotpotQA和ALFWorld的实验结果基于不同的主干模型。

    arXiv:2403.03101v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated great potential in complex reasoning tasks, yet they fall short when tackling more sophisticated challenges, especially when interacting with environments through generating executable actions. This inadequacy primarily stems from the lack of built-in action knowledge in language agents, which fails to effectively guide the planning trajectories during task solving and results in planning hallucination. To address this issue, we introduce KnowAgent, a novel approach designed to enhance the planning capabilities of LLMs by incorporating explicit action knowledge. Specifically, KnowAgent employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Experimental results on HotpotQA and ALFWorld based on various backbone m
    

