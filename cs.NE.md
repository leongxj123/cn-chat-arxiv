# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models.](http://arxiv.org/abs/2310.00194) | 这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。 |

# 详细

[^1]: 受前额叶皮层启发的大型语言模型规划架构

    A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models. (arXiv:2310.00194v1 [cs.AI])

    [http://arxiv.org/abs/2310.00194](http://arxiv.org/abs/2310.00194)

    这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。

    

    大型语言模型（LLM）在许多任务上展现出惊人的性能，但它们经常在需要多步推理或目标导向规划的任务中遇到困难。为了解决这个问题，我们从人脑中获取灵感，即通过前额叶皮层（PFC）中专门模块的重复交互来完成规划。这些模块执行冲突监测、状态预测、状态评估、任务分解和任务协调等功能。我们发现LLM有时能够单独执行这些功能，但在服务于一个目标时往往难以自主协调它们。因此，我们提出了一个带有多个基于LLM（GPT-4）模块的黑盒架构。该架构通过专门的PFC启发模块的交互将一个更大的问题分解为多个对LLM的简短自动调用，从而改善规划能力。我们在两个具有挑战性的规划任务上评估了组合架构。

    Large language models (LLMs) demonstrate impressive performance on a wide variety of tasks, but they often struggle with tasks that require multi-step reasoning or goal-directed planning. To address this, we take inspiration from the human brain, in which planning is accomplished via the recurrent interaction of specialized modules in the prefrontal cortex (PFC). These modules perform functions such as conflict monitoring, state prediction, state evaluation, task decomposition, and task coordination. We find that LLMs are sometimes capable of carrying out these functions in isolation, but struggle to autonomously coordinate them in the service of a goal. Therefore, we propose a black box architecture with multiple LLM-based (GPT-4) modules. The architecture improves planning through the interaction of specialized PFC-inspired modules that break down a larger problem into multiple brief automated calls to the LLM. We evaluate the combined architecture on two challenging planning tasks -
    

