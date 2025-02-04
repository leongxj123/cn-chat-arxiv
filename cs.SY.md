# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization](https://arxiv.org/abs/2311.18703) | 该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。 |
| [^2] | [DSAC-T: Distributional Soft Actor-Critic with Three Refinements.](http://arxiv.org/abs/2310.05858) | 本论文介绍了DSAC-T，通过评论者梯度调整、双值分布学习和基于方差的目标回报裁剪等三个改进对标准DSAC进行了改进，解决了标准DSAC存在的不稳定学习过程和对任务特定奖励缩放的问题，提高了算法的性能和适应性。 |

# 详细

[^1]: 通过熵率最小化实现可预测的强化学习动态

    Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization

    [https://arxiv.org/abs/2311.18703](https://arxiv.org/abs/2311.18703)

    该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。

    

    在强化学习中，智能体没有动机展示可预测的行为，通常通过策略熵正则化推动智能体在探索上随机化其行为。从人的角度来看，这使得强化学习智能体很难解释和预测；从安全角度来看，更难以进行形式化验证。我们提出了一种新的方法，称为可预测性感知强化学习（PA-RL），用于引导智能体展现可预测的行为，其利用状态序列熵率作为可预测性度量。我们展示了如何将熵率制定为平均奖励目标，并且由于其熵奖励函数依赖于策略，我们引入了一个动作相关的替代熵，以利用PG方法。我们证明了最小化平均替代奖励的确定性策略存在，并且最小化了实际熵率。我们还展示了如何在学习到的动态模型的基础上近似计算与值函数。

    In Reinforcement Learning (RL), agents have no incentive to exhibit predictable behaviors, and are often pushed (through e.g. policy entropy regularization) to randomize their actions in favor of exploration. From a human perspective, this makes RL agents hard to interpret and predict, and from a safety perspective, even harder to formally verify. We propose a novel method to induce predictable behavior in RL agents, referred to as Predictability-Aware RL (PA-RL), which employs the state sequence entropy rate as a predictability measure. We show how the entropy rate can be formulated as an average reward objective, and since its entropy reward function is policy-dependent, we introduce an action-dependent surrogate entropy enabling the use of PG methods. We prove that deterministic policies minimizing the average surrogate reward exist and also minimize the actual entropy rate, and show how, given a learned dynamical model, we are able to approximate the value function associated to th
    
[^2]: DSAC-T: 带有三个改进的分布式软角色扮演者—评论者

    DSAC-T: Distributional Soft Actor-Critic with Three Refinements. (arXiv:2310.05858v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.05858](http://arxiv.org/abs/2310.05858)

    本论文介绍了DSAC-T，通过评论者梯度调整、双值分布学习和基于方差的目标回报裁剪等三个改进对标准DSAC进行了改进，解决了标准DSAC存在的不稳定学习过程和对任务特定奖励缩放的问题，提高了算法的性能和适应性。

    

    强化学习在处理复杂的决策和控制任务方面已经被证明非常有效。然而，常见的无模型的强化学习方法往往面临严重的性能下降问题，这是由于众所周知的过估计问题所引起的。作为对这个问题的回应，我们最近引入了一种离线策略的强化学习算法，称为分布式软角色扮演者评论者（DSAC或DSAC-v1），它通过学习连续的高斯值分布来有效提高值估计的准确性。然而，标准DSAC也存在一些缺点，包括时而不稳定的学习过程和对任务特定的奖励缩放的需求，这可能会阻碍其在一些特殊任务中的整体性能和适应性。本文进一步引入了三个对标准DSAC的重要改进，以解决这些问题。这些改进包括评论者梯度调整、双值分布学习和基于方差的目标回报裁剪。修改后的强化学习算法称为DSAC-T。

    Reinforcement learning (RL) has proven to be highly effective in tackling complex decision-making and control tasks. However, prevalent model-free RL methods often face severe performance degradation due to the well-known overestimation issue. In response to this problem, we recently introduced an off-policy RL algorithm, called distributional soft actor-critic (DSAC or DSAC-v1), which can effectively improve the value estimation accuracy by learning a continuous Gaussian value distribution. Nonetheless, standard DSAC has its own shortcomings, including occasionally unstable learning processes and needs for task-specific reward scaling, which may hinder its overall performance and adaptability in some special tasks. This paper further introduces three important refinements to standard DSAC in order to address these shortcomings. These refinements consist of critic gradient adjusting, twin value distribution learning, and variance-based target return clipping. The modified RL algorithm 
    

