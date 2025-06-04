# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization](https://arxiv.org/abs/2311.18703) | 该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。 |

# 详细

[^1]: 通过熵率最小化实现可预测的强化学习动态

    Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization

    [https://arxiv.org/abs/2311.18703](https://arxiv.org/abs/2311.18703)

    该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。

    

    在强化学习中，智能体没有动机展示可预测的行为，通常通过策略熵正则化推动智能体在探索上随机化其行为。从人的角度来看，这使得强化学习智能体很难解释和预测；从安全角度来看，更难以进行形式化验证。我们提出了一种新的方法，称为可预测性感知强化学习（PA-RL），用于引导智能体展现可预测的行为，其利用状态序列熵率作为可预测性度量。我们展示了如何将熵率制定为平均奖励目标，并且由于其熵奖励函数依赖于策略，我们引入了一个动作相关的替代熵，以利用PG方法。我们证明了最小化平均替代奖励的确定性策略存在，并且最小化了实际熵率。我们还展示了如何在学习到的动态模型的基础上近似计算与值函数。

    In Reinforcement Learning (RL), agents have no incentive to exhibit predictable behaviors, and are often pushed (through e.g. policy entropy regularization) to randomize their actions in favor of exploration. From a human perspective, this makes RL agents hard to interpret and predict, and from a safety perspective, even harder to formally verify. We propose a novel method to induce predictable behavior in RL agents, referred to as Predictability-Aware RL (PA-RL), which employs the state sequence entropy rate as a predictability measure. We show how the entropy rate can be formulated as an average reward objective, and since its entropy reward function is policy-dependent, we introduce an action-dependent surrogate entropy enabling the use of PG methods. We prove that deterministic policies minimizing the average surrogate reward exist and also minimize the actual entropy rate, and show how, given a learned dynamical model, we are able to approximate the value function associated to th
    

