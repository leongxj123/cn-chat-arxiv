# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models](https://arxiv.org/abs/2403.17246) | 该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。 |

# 详细

[^1]: TwoStep: 使用经典规划器和大型语言模型进行多智能体任务规划

    TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models

    [https://arxiv.org/abs/2403.17246](https://arxiv.org/abs/2403.17246)

    该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。

    

    类似规划领域定义语言（PDDL）之类的经典规划公式允许确定可实现目标状态的动作序列，只要存在任何可能的初始状态。然而，PDDL中定义的推理问题并未捕获行动进行的时间方面，例如领域中的两个智能体如果彼此的后况不干扰前提条件，则可以同时执行一个动作。人类专家可以将目标分解为大部分独立的组成部分，并将每个智能体分配给其中一个子目标，以利用同时进行动作来加快计划步骤的执行，每个部分仅使用单个智能体规划。相比之下，直接推断计划步骤的大型语言模型（LLMs）并不保证执行成功，但利用常识推理来组装动作序列。我们通过近似人类直觉，结合了经典规划和LLMs的优势

    arXiv:2403.17246v1 Announce Type: new  Abstract: Classical planning formulations like the Planning Domain Definition Language (PDDL) admit action sequences guaranteed to achieve a goal state given an initial state if any are possible. However, reasoning problems defined in PDDL do not capture temporal aspects of action taking, for example that two agents in the domain can execute an action simultaneously if postconditions of each do not interfere with preconditions of the other. A human expert can decompose a goal into largely independent constituent parts and assign each agent to one of these subgoals to take advantage of simultaneous actions for faster execution of plan steps, each using only single agent planning. By contrast, large language models (LLMs) used for directly inferring plan steps do not guarantee execution success, but do leverage commonsense reasoning to assemble action sequences. We combine the strengths of classical planning and LLMs by approximating human intuition
    

