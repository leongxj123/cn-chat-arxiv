# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AgentMixer: Multi-Agent Correlated Policy Factorization.](http://arxiv.org/abs/2401.08728) | AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。 |

# 详细

[^1]: AgentMixer: 多智能体相关策略因子分解

    AgentMixer: Multi-Agent Correlated Policy Factorization. (arXiv:2401.08728v1 [cs.MA])

    [http://arxiv.org/abs/2401.08728](http://arxiv.org/abs/2401.08728)

    AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。

    

    集中式训练与分散式执行（CTDE）广泛应用于通过在训练过程中利用集中式值函数来稳定部分可观察的多智能体强化学习（MARL）。然而，现有方法通常假设智能体基于本地观测独立地做决策，这可能不会导致具有足够协调性的相关联的联合策略。受相关均衡概念的启发，我们提出引入"策略修改"来为智能体提供协调策略的机制。具体地，我们提出了一个新颖的框架AgentMixer，将联合完全可观测策略构造为各个部分可观测策略的非线性组合。为了实现分散式执行，可以通过模仿联合策略来得到各个部分策略。不幸的是，这种模仿学习可能会导致由于联合策略和个体策略之间的不匹配而导致的非对称学习失败。

    Centralized training with decentralized execution (CTDE) is widely employed to stabilize partially observable multi-agent reinforcement learning (MARL) by utilizing a centralized value function during training. However, existing methods typically assume that agents make decisions based on their local observations independently, which may not lead to a correlated joint policy with sufficient coordination. Inspired by the concept of correlated equilibrium, we propose to introduce a \textit{strategy modification} to provide a mechanism for agents to correlate their policies. Specifically, we present a novel framework, AgentMixer, which constructs the joint fully observable policy as a non-linear combination of individual partially observable policies. To enable decentralized execution, one can derive individual policies by imitating the joint policy. Unfortunately, such imitation learning can lead to \textit{asymmetric learning failure} caused by the mismatch between joint policy and indi
    

