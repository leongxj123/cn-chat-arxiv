# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents](https://arxiv.org/abs/2402.00798) | 本文提出了一种将自然语言和形式语言整合的“正式-LLM”框架，用于解决现有LLM智能体无法控制的计划生成问题。实验证明，该框架在提高生成计划性能和确保可控性方面取得了显著改进。 |

# 详细

[^1]: 正式-LLM：将形式语言和自然语言集成于可控的LLM智能体中

    Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents

    [https://arxiv.org/abs/2402.00798](https://arxiv.org/abs/2402.00798)

    本文提出了一种将自然语言和形式语言整合的“正式-LLM”框架，用于解决现有LLM智能体无法控制的计划生成问题。实验证明，该框架在提高生成计划性能和确保可控性方面取得了显著改进。

    

    最近，对于大型语言模型（LLMs）的进展使得人工智能智能体能够自动生成和执行解决复杂任务的多步计划。然而，由于LLM的内容生成过程几乎无法控制，当前的LLM智能体经常生成无效或不可执行的计划，这损害了生成计划的性能并破坏了用户对LLM智能体的信任。为应对这个问题，本文提出了一种新颖的“正式-LLM”框架，用于LLM智能体，通过将自然语言的表达力和形式语言的精确性进行整合。具体而言，该框架允许人类用户将他们对计划过程的要求或约束表达为自动机。然后，在自动机的监督下，使用基于堆栈的LLM计划生成过程来确保生成的计划满足约束条件，从而使计划过程可控。我们在基准任务和实际的真实任务上进行了实验，并且obtained significant improvements over existing LLM-based agents, demonstrating the effectiveness and controllability of the proposed Formal-LLM framework.

    Recent advancements on Large Language Models (LLMs) enable AI Agents to automatically generate and execute multi-step plans to solve complex tasks. However, since LLM's content generation process is hardly controllable, current LLM-based agents frequently generate invalid or non-executable plans, which jeopardizes the performance of the generated plans and corrupts users' trust in LLM-based agents. In response, this paper proposes a novel ``Formal-LLM'' framework for LLM-based agents by integrating the expressiveness of natural language and the precision of formal language. Specifically, the framework allows human users to express their requirements or constraints for the planning process as an automaton. A stack-based LLM plan generation process is then conducted under the supervision of the automaton to ensure that the generated plan satisfies the constraints, making the planning process controllable. We conduct experiments on both benchmark tasks and practical real-life tasks, and o
    

