# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inferring Latent Temporal Sparse Coordination Graph for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2403.19253) | 提出了一种用于多智能体强化学习的潜在时间稀疏协调图，能够有效处理智能体之间的协作关系并利用历史观测来进行知识交换 |

# 详细

[^1]: 推断多智能体强化学习的潜在时间稀疏协调图

    Inferring Latent Temporal Sparse Coordination Graph for Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2403.19253](https://arxiv.org/abs/2403.19253)

    提出了一种用于多智能体强化学习的潜在时间稀疏协调图，能够有效处理智能体之间的协作关系并利用历史观测来进行知识交换

    

    有效的智能体协调对于合作式多智能体强化学习(MARL)至关重要。当前MARL中的图学习方法局限性较大，仅仅依赖一步观察，忽略了重要的历史经验，导致生成的图存在缺陷，促进了冗余或有害信息交换。为了解决这些挑战，我们提出了推断用于MARL的潜在时间稀疏协调图（LTS-CG）。LTS-CG利用智能体的历史观测来计算智能体对概率矩阵，从中抽取稀疏图并用于智能体之间的知识交换，从而同时捕捉智能体的依赖关系和关系不确定性。该过程的计算复杂性仅与智能

    arXiv:2403.19253v1 Announce Type: new  Abstract: Effective agent coordination is crucial in cooperative Multi-Agent Reinforcement Learning (MARL). While agent cooperation can be represented by graph structures, prevailing graph learning methods in MARL are limited. They rely solely on one-step observations, neglecting crucial historical experiences, leading to deficient graphs that foster redundant or detrimental information exchanges. Additionally, high computational demands for action-pair calculations in dense graphs impede scalability. To address these challenges, we propose inferring a Latent Temporal Sparse Coordination Graph (LTS-CG) for MARL. The LTS-CG leverages agents' historical observations to calculate an agent-pair probability matrix, where a sparse graph is sampled from and used for knowledge exchange between agents, thereby simultaneously capturing agent dependencies and relation uncertainty. The computational complexity of this procedure is only related to the number o
    

