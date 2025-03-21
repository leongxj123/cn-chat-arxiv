# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2403.10996) | 提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题 |
| [^2] | [SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning](https://arxiv.org/abs/2401.16013) | 这个论文介绍了SERL软件套件，它是一个用于样本高效的机器人强化学习的库。该库包含了一个离线深度强化学习方法、计算奖励和重置环境的方法，高质量的机器人控制器，以及一些具有挑战性的示例任务。这个软件套件的目标是解决机器人强化学习的难以使用和获取性的挑战。 |
| [^3] | [Graphical Object-Centric Actor-Critic.](http://arxiv.org/abs/2310.17178) | 这项研究提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家和基于模型的方法结合起来，利用解耦的对象表示有效地学习策略。该方法填补了以对象为中心的强化学习环境中高效且适用于离散或连续动作空间的世界模型的研究空白。 |

# 详细

[^1]: 一个可扩展且可并行化的数字孪生框架，用于多智能体强化学习系统可持续Sim2Real转换

    A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2403.10996](https://arxiv.org/abs/2403.10996)

    提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题

    

    本工作提出了一个可持续的多智能体深度强化学习框架，能够选择性地按需扩展并行化训练工作负载，并利用最少的硬件资源将训练好的策略从模拟环境转移到现实世界。我们引入了AutoDRIVE生态系统作为一个启动数字孪生框架，用于训练、部署和转移合作和竞争的多智能体强化学习策略从模拟环境到现实世界。具体来说，我们首先探究了4台合作车辆(Nigel)在单智能体和多智能体学习环境中共享有限状态信息的交叉遍历问题，采用了一种通用策略方法。然后，我们使用个体策略方法研究了2辆车(F1TENTH)的对抗性自主赛车问题。在任何一组实验中，我们采用了去中心化学习架构，这允许对策略进行有力的训练和测试。

    arXiv:2403.10996v1 Announce Type: cross  Abstract: This work presents a sustainable multi-agent deep reinforcement learning framework capable of selectively scaling parallelized training workloads on-demand, and transferring the trained policies from simulation to reality using minimal hardware resources. We introduce AutoDRIVE Ecosystem as an enabling digital twin framework to train, deploy, and transfer cooperative as well as competitive multi-agent reinforcement learning policies from simulation to reality. Particularly, we first investigate an intersection traversal problem of 4 cooperative vehicles (Nigel) that share limited state information in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial autonomous racing problem of 2 vehicles (F1TENTH) using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted, which allowed robust training and testing of the policies 
    
[^2]: SERL: 用于样本高效的机器人强化学习的软件套件

    SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

    [https://arxiv.org/abs/2401.16013](https://arxiv.org/abs/2401.16013)

    这个论文介绍了SERL软件套件，它是一个用于样本高效的机器人强化学习的库。该库包含了一个离线深度强化学习方法、计算奖励和重置环境的方法，高质量的机器人控制器，以及一些具有挑战性的示例任务。这个软件套件的目标是解决机器人强化学习的难以使用和获取性的挑战。

    

    近年来，在机器人强化学习领域取得了显著进展，使得可以处理复杂的图像观察，实际训练，并结合辅助数据（如示范和先前经验）。然而，尽管取得了这些进展，机器人强化学习仍然难以使用。从实践者中认识到，这些算法的具体实现细节对性能的影响常常与算法选择同样重要（如果不是更重要）。我们认为，机器人强化学习被广泛采用以及进一步发展机器人强化学习方法的一个重要挑战是这些方法的相对难以获取性。为了解决这个挑战，我们开发了一个精心实现的库，其中包含了一种高效样本离线深度强化学习方法，以及计算奖励和重置环境的方法，针对广泛采用的机器人的高质量控制器，以及一些具有挑战性的示例任务。

    In recent years, significant progress has been made in the field of robotic reinforcement learning (RL), enabling methods that handle complex image observations, train in the real world, and incorporate auxiliary data, such as demonstrations and prior experience. However, despite these advances, robotic RL remains hard to use. It is acknowledged among practitioners that the particular implementation details of these algorithms are often just as important (if not more so) for performance as the choice of algorithm. We posit that a significant challenge to widespread adoption of robotic RL, as well as further development of robotic RL methods, is the comparative inaccessibility of such methods. To address this challenge, we developed a carefully implemented library containing a sample efficient off-policy deep RL method, together with methods for computing rewards and resetting the environment, a high-quality controller for a widely-adopted robot, and a number of challenging example task
    
[^3]: 图形化的以对象为中心的Actor-Critic算法

    Graphical Object-Centric Actor-Critic. (arXiv:2310.17178v1 [cs.AI])

    [http://arxiv.org/abs/2310.17178](http://arxiv.org/abs/2310.17178)

    这项研究提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家和基于模型的方法结合起来，利用解耦的对象表示有效地学习策略。该方法填补了以对象为中心的强化学习环境中高效且适用于离散或连续动作空间的世界模型的研究空白。

    

    最近在无监督的以对象为中心的表示学习及其在下游任务中的应用方面取得了重要进展。最新的研究支持这样一个观点，即在基于图像的以对象为中心的强化学习任务中采用解耦的对象表示能够促进策略学习。我们提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家算法和基于模型的方法结合起来，以有效利用这些表示。在我们的方法中，我们使用一个变换器编码器来提取对象表示，并使用图神经网络来近似环境的动力学。所提出的方法填补了开发强化学习环境中可以用于离散或连续动作空间的高效以对象为中心的世界模型的研究空白。我们的算法在一个具有复杂视觉3D机器人环境和一个具有组合结构的2D环境中表现更好。

    There have recently been significant advances in the problem of unsupervised object-centric representation learning and its application to downstream tasks. The latest works support the argument that employing disentangled object representations in image-based object-centric reinforcement learning tasks facilitates policy learning. We propose a novel object-centric reinforcement learning algorithm combining actor-critic and model-based approaches to utilize these representations effectively. In our approach, we use a transformer encoder to extract object representations and graph neural networks to approximate the dynamics of an environment. The proposed method fills a research gap in developing efficient object-centric world models for reinforcement learning settings that can be used for environments with discrete or continuous action spaces. Our algorithm performs better in a visually complex 3D robotic environment and a 2D environment with compositional structure than the state-of-t
    

