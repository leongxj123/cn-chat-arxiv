# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Models for Orchestrating Bimanual Robots](https://arxiv.org/abs/2404.02018) | 通过提出基于语言模型的双手编排（LABOR），本研究首次应对了在连续空间中进行双手任务协调的挑战。 |
| [^2] | [Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction](https://arxiv.org/abs/2402.15368) | 本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。 |

# 详细

[^1]: 用于编排双手机器人的大型语言模型

    Large Language Models for Orchestrating Bimanual Robots

    [https://arxiv.org/abs/2404.02018](https://arxiv.org/abs/2404.02018)

    通过提出基于语言模型的双手编排（LABOR），本研究首次应对了在连续空间中进行双手任务协调的挑战。

    

    尽管使机器人具有解决复杂操纵任务的能力已经取得了迅速进展，但为双手机器人生成控制策略以解决涉及两只手的任务仍然具有挑战性，原因是在有效的时间和空间协调方面存在困难。具有逐步推理和背景学习能力，大型语言模型（LLM）已经控制了各种机器人任务。然而，通过单个离散符号序列进行语言交流的本质使得LLM在连续空间中进行双手任务协调成为一项特殊挑战。为了首次通过LLM应对这一挑战，我们提出了基于语言模型的双手编排（LABOR），这是一个利用LLM分析任务配置并设计协调控制策略以解决长期双手任务的代理。在模拟环境中，LABOR代理进行了评估。

    arXiv:2404.02018v1 Announce Type: cross  Abstract: Although there has been rapid progress in endowing robots with the ability to solve complex manipulation tasks, generating control policies for bimanual robots to solve tasks involving two hands is still challenging because of the difficulties in effective temporal and spatial coordination. With emergent abilities in terms of step-by-step reasoning and in-context learning, Large Language Models (LLMs) have taken control of a variety of robotic tasks. However, the nature of language communication via a single sequence of discrete symbols makes LLM-based coordination in continuous space a particular challenge for bimanual tasks. To tackle this challenge for the first time by an LLM, we present LAnguage-model-based Bimanual ORchestration (LABOR), an agent utilizing an LLM to analyze task configurations and devise coordination control policies for addressing long-horizon bimanual tasks. In the simulated environment, the LABOR agent is eval
    
[^2]: 使用符合预测的技术实现语言指导多机器人系统的安全任务规划

    Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction

    [https://arxiv.org/abs/2402.15368](https://arxiv.org/abs/2402.15368)

    本文引入了一种新的基于分布式LLM和符合预测技术的多机器人规划器，实现了高任务成功率。

    

    本文解决了语言指导机器人团队的任务规划问题。任务用自然语言（NL）表示，要求机器人在各种位置和语义对象上应用它们的能力（例如移动、操作和感知）。最近几篇论文通过利用预训练的大型语言模型（LLMs）设计有效的多机器人计划来解决类似的规划问题。然而，这些方法缺乏任务性能和安全性保证。为了解决这一挑战，我们引入了一种新的基于分布式LLM的规划器，能够实现高任务成功率。这是通过利用符合预测（CP）来实现的，CP是一种基于分布的不确定性量化工具，可以在黑盒模型中对其固有不确定性进行推理。CP允许所提出的多机器人规划器以分布方式推理其固有不确定性，使得机器人在充分信任时能够做出个别决策。

    arXiv:2402.15368v1 Announce Type: cross  Abstract: This paper addresses task planning problems for language-instructed robot teams. Tasks are expressed in natural language (NL), requiring the robots to apply their capabilities (e.g., mobility, manipulation, and sensing) at various locations and semantic objects. Several recent works have addressed similar planning problems by leveraging pre-trained Large Language Models (LLMs) to design effective multi-robot plans. However, these approaches lack mission performance and safety guarantees. To address this challenge, we introduce a new decentralized LLM-based planner that is capable of achieving high mission success rates. This is accomplished by leveraging conformal prediction (CP), a distribution-free uncertainty quantification tool in black-box models. CP allows the proposed multi-robot planner to reason about its inherent uncertainty in a decentralized fashion, enabling robots to make individual decisions when they are sufficiently ce
    

