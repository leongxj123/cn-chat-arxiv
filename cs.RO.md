# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods](https://arxiv.org/abs/2404.00282) | 大型语言模型在强化学习中具有潜在优势，通过结构化分类和角色分析，为未来研究提供指导。 |
| [^2] | [D3G: Learning Multi-robot Coordination from Demonstrations.](http://arxiv.org/abs/2207.08892) | 本文提出了一个D3G框架，可以从演示中学习多机器人协调。通过最小化轨迹与演示之间的不匹配，每个机器人可以自动调整其个体动态和目标，提高了学习效率和效果。 |

# 详细

[^1]: 基于大型语言模型增强强化学习的调查:概念、分类和方法

    Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods

    [https://arxiv.org/abs/2404.00282](https://arxiv.org/abs/2404.00282)

    大型语言模型在强化学习中具有潜在优势，通过结构化分类和角色分析，为未来研究提供指导。

    

    随着大规模语言模型(LLMs)拥有广泛的预训练知识和高级通用能力，它们在增强学习方面如多任务学习、样本效率和任务规划等方面展现出潜力。本调查综述了现有$\textit{LLM增强RL}$文献，总结了其与传统RL方法的特征，旨在澄清研究范围和未来研究方向。利用经典的Agent-环境交互范例，我们提出了一个结构化的分类法，系统地将LLMs在RL中的功能分类，包括四种角色：信息处理器、奖励设计者、决策者和生成器。此外，针对每个角色，我们总结了方法论，分析了缓解的特定RL挑战，并提供了未来方向的见解。最后，潜在应用、前景

    arXiv:2404.00282v1 Announce Type: cross  Abstract: With extensive pre-trained knowledge and high-level general capabilities, large language models (LLMs) emerge as a promising avenue to augment reinforcement learning (RL) in aspects such as multi-task learning, sample efficiency, and task planning. In this survey, we provide a comprehensive review of the existing literature in $\textit{LLM-enhanced RL}$ and summarize its characteristics compared to conventional RL methods, aiming to clarify the research scope and directions for future studies. Utilizing the classical agent-environment interaction paradigm, we propose a structured taxonomy to systematically categorize LLMs' functionalities in RL, including four roles: information processor, reward designer, decision-maker, and generator. Additionally, for each role, we summarize the methodologies, analyze the specific RL challenges that are mitigated, and provide insights into future directions. Lastly, potential applications, prospecti
    
[^2]: D3G: 从演示中学习多机器人协调

    D3G: Learning Multi-robot Coordination from Demonstrations. (arXiv:2207.08892v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2207.08892](http://arxiv.org/abs/2207.08892)

    本文提出了一个D3G框架，可以从演示中学习多机器人协调。通过最小化轨迹与演示之间的不匹配，每个机器人可以自动调整其个体动态和目标，提高了学习效率和效果。

    

    本文开发了一个分布式可微动态游戏（D3G）框架，可以实现从演示中学习多机器人协调。我们将多机器人协调表示为一个动态游戏，其中一个机器人的行为受其自身动态和目标的控制，同时也取决于其他机器人的行为。因此，通过调整每个机器人的目标和动态，可以适应协调。所提出的D3G使每个机器人通过最小化其轨迹与演示之间的不匹配，在分布式方式下自动调整其个体动态和目标。该学习框架具有新的设计，包括一个前向传递，所有机器人合作寻找游戏的纳什均衡，以及一个反向传递，在通信图中传播梯度。我们在仿真中测试了D3G，并给出了不同任务配置的两种机器人。结果证明了D3G学习多机器人协调的能力。

    This paper develops a Distributed Differentiable Dynamic Game (D3G) framework, which enables learning multi-robot coordination from demonstrations. We represent multi-robot coordination as a dynamic game, where the behavior of a robot is dictated by its own dynamics and objective that also depends on others' behavior. The coordination thus can be adapted by tuning the objective and dynamics of each robot. The proposed D3G enables each robot to automatically tune its individual dynamics and objectives in a distributed manner by minimizing the mismatch between its trajectory and demonstrations. This learning framework features a new design, including a forward-pass, where all robots collaboratively seek Nash equilibrium of a game, and a backward-pass, where gradients are propagated via the communication graph. We test the D3G in simulation with two types of robots given different task configurations. The results validate the capability of D3G for learning multi-robot coordination from de
    

