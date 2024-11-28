# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint](https://arxiv.org/abs/2402.14244) | 使用人类反馈和动态距离约束对层次化强化学习进行引导，解决了找到适当子目标的问题，并设计了双策略以稳定训练。 |

# 详细

[^1]: MENTOR：在层次化强化学习中引导人类反馈和动态距离约束

    MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint

    [https://arxiv.org/abs/2402.14244](https://arxiv.org/abs/2402.14244)

    使用人类反馈和动态距离约束对层次化强化学习进行引导，解决了找到适当子目标的问题，并设计了双策略以稳定训练。

    

    层次化强化学习（HRL）为智能体的复杂任务提供了一种有前途的解决方案，其中使用了将任务分解为子目标并依次完成的层次框架。然而，当前的方法难以找到适当的子目标来确保稳定的学习过程。为了解决这个问题，我们提出了一个通用的层次强化学习框架，将人类反馈和动态距离约束整合到其中（MENTOR）。MENTOR充当“导师”，将人类反馈纳入高层策略学习中，以找到更好的子目标。至于低层策略，MENTOR设计了一个双策略以分别进行探索-开发解耦，以稳定训练。此外，尽管人类可以简单地将任务拆分成...

    arXiv:2402.14244v1 Announce Type: new  Abstract: Hierarchical reinforcement learning (HRL) provides a promising solution for complex tasks with sparse rewards of intelligent agents, which uses a hierarchical framework that divides tasks into subgoals and completes them sequentially. However, current methods struggle to find suitable subgoals for ensuring a stable learning process. Without additional guidance, it is impractical to rely solely on exploration or heuristics methods to determine subgoals in a large goal space. To address the issue, We propose a general hierarchical reinforcement learning framework incorporating human feedback and dynamic distance constraints (MENTOR). MENTOR acts as a "mentor", incorporating human feedback into high-level policy learning, to find better subgoals. As for low-level policy, MENTOR designs a dual policy for exploration-exploitation decoupling respectively to stabilize the training. Furthermore, although humans can simply break down tasks into s
    

