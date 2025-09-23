# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Asynchronous Perception-Action-Communication with Graph Neural Networks.](http://arxiv.org/abs/2309.10164) | 该论文提出了使用图神经网络实现异步感知-动作-通信的方法，解决了在大型机器人群体中协作和通信的挑战。现有的框架假设顺序执行，该方法是完全分散的，但在评估和部署方面仍存在一些限制。 |
| [^2] | [Exploiting Symmetry and Heuristic Demonstrations in Off-policy Reinforcement Learning for Robotic Manipulation.](http://arxiv.org/abs/2304.06055) | 本文提出了一个离线强化学习方法，该方法利用对称环境中的专家演示来进行机器人操作的策略训练，从而提高了学习效率和样本效率。 |

# 详细

[^1]: 异步感知-动作-通信与图神经网络

    Asynchronous Perception-Action-Communication with Graph Neural Networks. (arXiv:2309.10164v1 [cs.RO])

    [http://arxiv.org/abs/2309.10164](http://arxiv.org/abs/2309.10164)

    该论文提出了使用图神经网络实现异步感知-动作-通信的方法，解决了在大型机器人群体中协作和通信的挑战。现有的框架假设顺序执行，该方法是完全分散的，但在评估和部署方面仍存在一些限制。

    

    在大型机器人群体中实现共同的全局目标的协作是一个具有挑战性的问题，因为机器人的感知和通信能力有限。机器人必须执行感知-动作-通信（PAC）循环-它们感知局部环境，与其他机器人通信，并实时采取行动。分散的PAC系统面临的一个基本挑战是决定与相邻机器人通信的信息以及如何在利用邻居共享的信息的同时采取行动。最近，使用图神经网络（GNNs）来解决这个问题已经取得了一些进展，比如在群集和覆盖控制等应用中。虽然在概念上，GNN策略是完全分散的，但评估和部署这样的策略主要仍然是集中式的或具有限制性的分散式。此外，现有的框架假设感知和动作推理的顺序执行，这在现实世界的应用中非常限制性。

    Collaboration in large robot swarms to achieve a common global objective is a challenging problem in large environments due to limited sensing and communication capabilities. The robots must execute a Perception-Action-Communication (PAC) loop -- they perceive their local environment, communicate with other robots, and take actions in real time. A fundamental challenge in decentralized PAC systems is to decide what information to communicate with the neighboring robots and how to take actions while utilizing the information shared by the neighbors. Recently, this has been addressed using Graph Neural Networks (GNNs) for applications such as flocking and coverage control. Although conceptually, GNN policies are fully decentralized, the evaluation and deployment of such policies have primarily remained centralized or restrictively decentralized. Furthermore, existing frameworks assume sequential execution of perception and action inference, which is very restrictive in real-world applica
    
[^2]: 利用对称性和启发式演示来进行机器人操作的离线强化学习

    Exploiting Symmetry and Heuristic Demonstrations in Off-policy Reinforcement Learning for Robotic Manipulation. (arXiv:2304.06055v1 [cs.RO])

    [http://arxiv.org/abs/2304.06055](http://arxiv.org/abs/2304.06055)

    本文提出了一个离线强化学习方法，该方法利用对称环境中的专家演示来进行机器人操作的策略训练，从而提高了学习效率和样本效率。

    

    强化学习在许多领域中自动构建控制策略具有显著潜力，但在应用于机器人操作任务时由于维度的问题，效率较低。为了促进这些任务的学习，先前的知识或启发式方法可以有效地提高学习性能。本文旨在定义和结合物理机器环境中存在的自然对称性，利用对称环境中的专家演示通过强化学习和行为克隆的融合来训练具有高样本效率的策略，从而给离线强化学习过程提供多样化而紧凑的启动。此外，本文提出了一个最近概念的严格框架，并探索了它在机器人操作任务中的范围。该方法通过在模拟环境中进行两个点对点的工业臂到达任务（有障碍和无障碍）的验证。

    Reinforcement learning demonstrates significant potential in automatically building control policies in numerous domains, but shows low efficiency when applied to robot manipulation tasks due to the curse of dimensionality. To facilitate the learning of such tasks, prior knowledge or heuristics that incorporate inherent simplification can effectively improve the learning performance. This paper aims to define and incorporate the natural symmetry present in physical robotic environments. Then, sample-efficient policies are trained by exploiting the expert demonstrations in symmetrical environments through an amalgamation of reinforcement and behavior cloning, which gives the off-policy learning process a diverse yet compact initiation. Furthermore, it presents a rigorous framework for a recent concept and explores its scope for robot manipulation tasks. The proposed method is validated via two point-to-point reaching tasks of an industrial arm, with and without an obstacle, in a simulat
    

