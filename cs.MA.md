# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AgentMixer: Multi-Agent Correlated Policy Factorization.](http://arxiv.org/abs/2401.08728) | AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。 |
| [^2] | [Maximum Entropy Heterogeneous-Agent Mirror Learning.](http://arxiv.org/abs/2306.10715) | 最大熵异质代理镜像学习(MEHAML)是一种新的理论框架，通过最大熵原理设计了最大熵MARL的演员-评论家算法，具有联合最大熵目标的单调改进和收敛至中位响应均衡(QRE)的期望特性，并通过扩展常用的强化学习算法HASAC来验证其实用性和在探索和稳健性方面的显著改进。 |

# 详细

[^1]: AgentMixer: 多智能体相关策略因子分解

    AgentMixer: Multi-Agent Correlated Policy Factorization. (arXiv:2401.08728v1 [cs.MA])

    [http://arxiv.org/abs/2401.08728](http://arxiv.org/abs/2401.08728)

    AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。

    

    集中式训练与分散式执行（CTDE）广泛应用于通过在训练过程中利用集中式值函数来稳定部分可观察的多智能体强化学习（MARL）。然而，现有方法通常假设智能体基于本地观测独立地做决策，这可能不会导致具有足够协调性的相关联的联合策略。受相关均衡概念的启发，我们提出引入"策略修改"来为智能体提供协调策略的机制。具体地，我们提出了一个新颖的框架AgentMixer，将联合完全可观测策略构造为各个部分可观测策略的非线性组合。为了实现分散式执行，可以通过模仿联合策略来得到各个部分策略。不幸的是，这种模仿学习可能会导致由于联合策略和个体策略之间的不匹配而导致的非对称学习失败。

    Centralized training with decentralized execution (CTDE) is widely employed to stabilize partially observable multi-agent reinforcement learning (MARL) by utilizing a centralized value function during training. However, existing methods typically assume that agents make decisions based on their local observations independently, which may not lead to a correlated joint policy with sufficient coordination. Inspired by the concept of correlated equilibrium, we propose to introduce a \textit{strategy modification} to provide a mechanism for agents to correlate their policies. Specifically, we present a novel framework, AgentMixer, which constructs the joint fully observable policy as a non-linear combination of individual partially observable policies. To enable decentralized execution, one can derive individual policies by imitating the joint policy. Unfortunately, such imitation learning can lead to \textit{asymmetric learning failure} caused by the mismatch between joint policy and indi
    
[^2]: 最大熵异质代理镜像学习

    Maximum Entropy Heterogeneous-Agent Mirror Learning. (arXiv:2306.10715v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2306.10715](http://arxiv.org/abs/2306.10715)

    最大熵异质代理镜像学习(MEHAML)是一种新的理论框架，通过最大熵原理设计了最大熵MARL的演员-评论家算法，具有联合最大熵目标的单调改进和收敛至中位响应均衡(QRE)的期望特性，并通过扩展常用的强化学习算法HASAC来验证其实用性和在探索和稳健性方面的显著改进。

    

    多智能体强化学习(MARL)在合作博弈中表现出有效性。然而，现有的最先进方法面临样本效率低、超参数脆弱性和收敛于次优纳什均衡的风险等挑战。为了解决这些问题，本文提出了一种新的理论框架，命名为最大熵异质代理镜像学习(MEHAML)，利用最大熵原理设计了最大熵MARL的演员-评论家算法。我们证明了从MEHAML框架导出的算法具有联合最大熵目标的单调改进和收敛至中位响应均衡(QRE)的期望特性。MEHAML的实用性通过开发广泛使用的强化学习算法HASAC的MEHAML扩展来展示，在三个具有挑战性的基准测试上展示出了探索和稳健性的显著提升。

    Multi-agent reinforcement learning (MARL) has been shown effective for cooperative games in recent years. However, existing state-of-the-art methods face challenges related to sample inefficiency, brittleness regarding hyperparameters, and the risk of converging to a suboptimal Nash Equilibrium. To resolve these issues, in this paper, we propose a novel theoretical framework, named Maximum Entropy Heterogeneous-Agent Mirror Learning (MEHAML), that leverages the maximum entropy principle to design maximum entropy MARL actor-critic algorithms. We prove that algorithms derived from the MEHAML framework enjoy the desired properties of the monotonic improvement of the joint maximum entropy objective and the convergence to quantal response equilibrium (QRE). The practicality of MEHAML is demonstrated by developing a MEHAML extension of the widely used RL algorithm, HASAC (for soft actor-critic), which shows significant improvements in exploration and robustness on three challenging benchmark
    

