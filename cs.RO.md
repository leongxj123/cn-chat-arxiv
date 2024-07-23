# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Socially Integrated Navigation: A Social Acting Robot with Deep Reinforcement Learning](https://arxiv.org/abs/2403.09793) | 提出了一种新颖的社会整合导航方法，通过与人的互动使机器人的社交行为自适应，并从其他基于DRL的导航方法中区分出具有明确预定义社交行为的社会意识方法和缺乏社交行为的社会碰撞回避。 |
| [^2] | [Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion.](http://arxiv.org/abs/2308.12517) | 本文提出了一种新的强化学习框架，为复杂机器人系统训练神经网络控制器。该框架引入了奖励和约束的概念，通过设计高效的策略优化算法来处理约束，以减少计算开销。通过应用于不同腿式机器人的运动控制器训练中，展示了该框架的有效性。 |
| [^3] | [Improving Offline-to-Online Reinforcement Learning with Q-Ensembles.](http://arxiv.org/abs/2306.06871) | 我们提出了一种名为Q-Ensembles的新框架，通过增加Q网络的数量，无缝地连接离线预训练和在线微调，同时不降低性能。此外，我们适当放宽Q值估计的悲观性，并将基于集合的探索机制融入我们的框架中，从而提升了离线到在线强化学习的性能。 |

# 详细

[^1]: 社会整合导航：具有深度强化学习的社交行动机器人

    Socially Integrated Navigation: A Social Acting Robot with Deep Reinforcement Learning

    [https://arxiv.org/abs/2403.09793](https://arxiv.org/abs/2403.09793)

    提出了一种新颖的社会整合导航方法，通过与人的互动使机器人的社交行为自适应，并从其他基于DRL的导航方法中区分出具有明确预定义社交行为的社会意识方法和缺乏社交行为的社会碰撞回避。

    

    移动机器人正在广泛应用于各种拥挤场景，并成为我们社会的一部分。一个具有个体人类考虑的社会可接受的导航行为对于可扩展的应用和人类接受至关重要。最近使用深度强化学习（DRL）方法来学习机器人的导航策略，并对机器人与人类之间的复杂交互进行建模。我们建议根据机器人展示的社交行为将现有基于DRL的导航方法分为具有缺乏社交行为的社会碰撞回避和具有明确预定义社交行为的社会意识方法。此外，我们提出了一种新颖的社会整合导航方法，其中机器人的社交行为是自适应的，并且是通过与人类的互动而产生的。我们的方法的构式源自社会学定义，

    arXiv:2403.09793v1 Announce Type: cross  Abstract: Mobile robots are being used on a large scale in various crowded situations and become part of our society. The socially acceptable navigation behavior of a mobile robot with individual human consideration is an essential requirement for scalable applications and human acceptance. Deep Reinforcement Learning (DRL) approaches are recently used to learn a robot's navigation policy and to model the complex interactions between robots and humans. We propose to divide existing DRL-based navigation approaches based on the robot's exhibited social behavior and distinguish between social collision avoidance with a lack of social behavior and socially aware approaches with explicit predefined social behavior. In addition, we propose a novel socially integrated navigation approach where the robot's social behavior is adaptive and emerges from the interaction with humans. The formulation of our approach is derived from a sociological definition, 
    
[^2]: 不仅仅奖励，还有约束：用于腿式机器人运动的应用

    Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion. (arXiv:2308.12517v1 [cs.RO])

    [http://arxiv.org/abs/2308.12517](http://arxiv.org/abs/2308.12517)

    本文提出了一种新的强化学习框架，为复杂机器人系统训练神经网络控制器。该框架引入了奖励和约束的概念，通过设计高效的策略优化算法来处理约束，以减少计算开销。通过应用于不同腿式机器人的运动控制器训练中，展示了该框架的有效性。

    

    早期的一些研究通过设计神经网络控制器并使用无模型强化学习来训练，展示了复杂机器人系统中令人印象深刻的控制性能。然而，这些具有自然动作风格和高任务性能的出色控制器是通过进行大量奖励工程而开发的，该过程非常费时费力，需要设计大量奖励项并确定合适的奖励系数。在这项工作中，我们提出了一种新的强化学习框架，用于训练同时包含奖励和约束的神经网络控制器。为了让工程师能够适当地反映他们对约束的意图并以最小的计算开销处理它们，我们提出了两种约束类型和一种高效的策略优化算法。该学习框架被应用于训练不同形态和物理属性的几个腿式机器人的运动控制器。

    Several earlier studies have shown impressive control performance in complex robotic systems by designing the controller using a neural network and training it with model-free reinforcement learning. However, these outstanding controllers with natural motion style and high task performance are developed through extensive reward engineering, which is a highly laborious and time-consuming process of designing numerous reward terms and determining suitable reward coefficients. In this work, we propose a novel reinforcement learning framework for training neural network controllers for complex robotic systems consisting of both rewards and constraints. To let the engineers appropriately reflect their intent to constraints and handle them with minimal computation overhead, two constraint types and an efficient policy optimization algorithm are suggested. The learning framework is applied to train locomotion controllers for several legged robots with different morphology and physical attribu
    
[^3]: 提升离线到在线强化学习的Q-Ensembles方法

    Improving Offline-to-Online Reinforcement Learning with Q-Ensembles. (arXiv:2306.06871v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.06871](http://arxiv.org/abs/2306.06871)

    我们提出了一种名为Q-Ensembles的新框架，通过增加Q网络的数量，无缝地连接离线预训练和在线微调，同时不降低性能。此外，我们适当放宽Q值估计的悲观性，并将基于集合的探索机制融入我们的框架中，从而提升了离线到在线强化学习的性能。

    

    离线强化学习是一种学习范式，代理根据固定的经验数据集进行学习。然而，仅从静态数据集中学习可能限制了性能，因为缺乏探索能力。为了克服这个问题，将离线预训练与在线微调结合起来的离线到在线强化学习方法能够让代理与环境实时交互，进一步完善其策略。然而，现有的离线到在线强化学习方法存在性能下降和在线阶段改进缓慢的问题。为了解决这些挑战，我们提出了一种名为Q-Ensembles的新框架，它通过增加Q网络的数量，无缝地连接离线预训练和在线微调，同时不降低性能。此外，为了加快在线性能提升，我们适当放宽Q值估计的悲观性，并将基于集合的探索机制融入我们的框架中。

    Offline reinforcement learning (RL) is a learning paradigm where an agent learns from a fixed dataset of experience. However, learning solely from a static dataset can limit the performance due to the lack of exploration. To overcome it, offline-to-online RL combines offline pre-training with online fine-tuning, which enables the agent to further refine its policy by interacting with the environment in real-time. Despite its benefits, existing offline-to-online RL methods suffer from performance degradation and slow improvement during the online phase. To tackle these challenges, we propose a novel framework called Ensemble-based Offline-to-Online (E2O) RL. By increasing the number of Q-networks, we seamlessly bridge offline pre-training and online fine-tuning without degrading performance. Moreover, to expedite online performance enhancement, we appropriately loosen the pessimism of Q-value estimation and incorporate ensemble-based exploration mechanisms into our framework. Experiment
    

