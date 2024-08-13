# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conditional Logical Message Passing Transformer for Complex Query Answering](https://arxiv.org/abs/2402.12954) | 提出了一种考虑查询图中常量和变量之间差异，能动态测量消息重要性并捕捉隐式逻辑依赖关系的条件逻辑消息传递变压器。 |

# 详细

[^1]: 用于复杂查询回答的条件逻辑消息传递变压器

    Conditional Logical Message Passing Transformer for Complex Query Answering

    [https://arxiv.org/abs/2402.12954](https://arxiv.org/abs/2402.12954)

    提出了一种考虑查询图中常量和变量之间差异，能动态测量消息重要性并捕捉隐式逻辑依赖关系的条件逻辑消息传递变压器。

    

    知识图谱（KGs）上的复杂查询回答（CQA）是一项具有挑战性的任务。由于KGs通常是不完整的，提出了神经模型来通过执行多跳逻辑推理来解决CQA。然而，大多数模型不能同时在一跳和多跳查询上表现良好。最近的工作提出了一种基于预训练神经链接预测器的逻辑消息传递机制。虽然在一跳和多跳查询上都有效，但它忽略了查询图中常量和变量节点之间的差异。此外，在节点嵌入更新阶段，该机制不能动态衡量不同消息的重要性，并且它能否捕捉与节点和接收消息相关的隐式逻辑依赖关系仍不清楚。在本文中，我们提出了条件逻辑消息传递变压器（CLMPT），考虑了查询图中常量和变量之间的差异，并且具有动态测量不同消息重要性以及捕捉与节点和接收消息相关的隐式逻辑依赖关系的能力。

    arXiv:2402.12954v1 Announce Type: cross  Abstract: Complex Query Answering (CQA) over Knowledge Graphs (KGs) is a challenging task. Given that KGs are usually incomplete, neural models are proposed to solve CQA by performing multi-hop logical reasoning. However, most of them cannot perform well on both one-hop and multi-hop queries simultaneously. Recent work proposes a logical message passing mechanism based on the pre-trained neural link predictors. While effective on both one-hop and multi-hop queries, it ignores the difference between the constant and variable nodes in a query graph. In addition, during the node embedding update stage, this mechanism cannot dynamically measure the importance of different messages, and whether it can capture the implicit logical dependencies related to a node and received messages remains unclear. In this paper, we propose Conditional Logical Message Passing Transformer (CLMPT), which considers the difference between constants and variables in the c
    

