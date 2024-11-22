# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction](https://arxiv.org/abs/2402.15368) | 本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。 |

# 详细

[^1]: 使用符合预测的技术实现语言指导多机器人系统的安全任务规划

    Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction

    [https://arxiv.org/abs/2402.15368](https://arxiv.org/abs/2402.15368)

    本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。

    

    本文解决了语言指导机器人团队的任务规划问题。任务用自然语言（NL）表示，要求机器人在各种位置和语义对象上应用它们的能力（例如移动、操作和感知）。最近几篇论文通过利用预训练的大型语言模型（LLMs）设计有效的多机器人计划来解决类似的规划问题。然而，这些方法缺乏任务性能和安全性保证。为了解决这一挑战，我们引入了一种新的基于分布式LLM的规划器，能够实现高任务成功率。这是通过利用符合预测（CP）来实现的，CP是一种基于分布的不确定性量化工具，可以在黑盒模型中对其固有不确定性进行推理。CP允许所提出的多机器人规划器以分布方式推理其固有不确定性，使得机器人在充分信任时能够做出个别决策。

    arXiv:2402.15368v1 Announce Type: cross  Abstract: This paper addresses task planning problems for language-instructed robot teams. Tasks are expressed in natural language (NL), requiring the robots to apply their capabilities (e.g., mobility, manipulation, and sensing) at various locations and semantic objects. Several recent works have addressed similar planning problems by leveraging pre-trained Large Language Models (LLMs) to design effective multi-robot plans. However, these approaches lack mission performance and safety guarantees. To address this challenge, we introduce a new decentralized LLM-based planner that is capable of achieving high mission success rates. This is accomplished by leveraging conformal prediction (CP), a distribution-free uncertainty quantification tool in black-box models. CP allows the proposed multi-robot planner to reason about its inherent uncertainty in a decentralized fashion, enabling robots to make individual decisions when they are sufficiently ce
    

