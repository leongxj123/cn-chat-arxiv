# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight](https://arxiv.org/abs/2403.12203) | 在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调 |
| [^2] | [UAV Pathfinding in Dynamic Obstacle Avoidance with Multi-agent Reinforcement Learning.](http://arxiv.org/abs/2310.16659) | 本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。实验结果验证了该方法的有效性。 |
| [^3] | [Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects.](http://arxiv.org/abs/2208.04883) | 本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。 |

# 详细

[^1]: 基于模仿的增强学习为基于视觉的敏捷飞行引导引导

    Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight

    [https://arxiv.org/abs/2403.12203](https://arxiv.org/abs/2403.12203)

    在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调

    

    我们在基于视觉的自主无人机竞速的背景下，将强化学习（RL）的有效性和模仿学习（IL）的效率结合在一起。我们专注于直接处理视觉输入，而无需明确的状态估计。虽然强化学习通过试错提供了一个学习复杂控制器的通用框架，但面临着样本效率和计算需求的挑战，因为视觉输入的维度较高。相反，IL在从视觉演示中学习方面表现出效率，但受到演示质量的限制，并面临诸如协变量漂移的问题。为了克服这些限制，我们提出了一个结合RL和IL优势的新型训练框架。我们的框架包括三个阶段：使用特权状态信息的师傅策略的初始训练，使用IL将此策略蒸馏为学生策略，以及性能受限的自适应RL微调

    arXiv:2403.12203v1 Announce Type: cross  Abstract: We combine the effectiveness of Reinforcement Learning (RL) and the efficiency of Imitation Learning (IL) in the context of vision-based, autonomous drone racing. We focus on directly processing visual input without explicit state estimation. While RL offers a general framework for learning complex controllers through trial and error, it faces challenges regarding sample efficiency and computational demands due to the high dimensionality of visual inputs. Conversely, IL demonstrates efficiency in learning from visual demonstrations but is limited by the quality of those demonstrations and faces issues like covariate shift. To overcome these limitations, we propose a novel training framework combining RL and IL's advantages. Our framework involves three stages: initial training of a teacher policy using privileged state information, distilling this policy into a student policy using IL, and performance-constrained adaptive RL fine-tunin
    
[^2]: 无人机在多智能体强化学习中的动态避障路径规划

    UAV Pathfinding in Dynamic Obstacle Avoidance with Multi-agent Reinforcement Learning. (arXiv:2310.16659v1 [cs.RO])

    [http://arxiv.org/abs/2310.16659](http://arxiv.org/abs/2310.16659)

    本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。实验结果验证了该方法的有效性。

    

    多智能体强化学习方法在动态和不确定的场景中在线规划智能体的可行且安全的路径具有重要意义。本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。在该方法中，每个智能体仅与中央规划者或其邻居进行通信，以在线规划可行且安全的路径。我们基于模型预测控制的思想改进了我们的方法，以提高智能体的训练效率和采样利用率。在模拟、室内和室外环境中的实验结果验证了我们方法的有效性。

    Multi-agent reinforcement learning based methods are significant for online planning of feasible and safe paths for agents in dynamic and uncertain scenarios. Although some methods like fully centralized and fully decentralized methods achieve a certain measure of success, they also encounter problems such as dimension explosion and poor convergence, respectively. In this paper, we propose a novel centralized training with decentralized execution method based on multi-agent reinforcement learning to solve the dynamic obstacle avoidance problem online. In this approach, each agent communicates only with the central planner or only with its neighbors, respectively, to plan feasible and safe paths online. We improve our methods based on the idea of model predictive control to increase the training efficiency and sample utilization of agents. The experimental results in both simulation, indoor, and outdoor environments validate the effectiveness of our method. The video is available at htt
    
[^3]: 神经会合：面向星际物体的可靠导航和控制的证明

    Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects. (arXiv:2208.04883v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2208.04883](http://arxiv.org/abs/2208.04883)

    本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。

    

    星际物体（ISOs）很可能是不可替代的原始材料，在理解系外行星星系方面具有重要价值。然而，由于其运行轨道难以约束，通常具有较高的倾角和相对速度，使用传统的人在环路方法探索ISOs具有相当大的挑战性。本文提出了一种名为神经会合的深度学习导航和控制框架，用于在实时中以可靠、准确和自主的方式遭遇快速移动的物体，包括ISOs。它在基于谱归一化的深度神经网络的引导策略之上使用点最小范数追踪控制，其中参数通过直接惩罚MPC状态轨迹跟踪误差的损失函数进行调优。我们展示了神经会合在预期的飞行器交付误差上提供了高概率指数上界，其证明利用了随机递增稳定性分析。

    Interstellar objects (ISOs) are likely representatives of primitive materials invaluable in understanding exoplanetary star systems. Due to their poorly constrained orbits with generally high inclinations and relative velocities, however, exploring ISOs with conventional human-in-the-loop approaches is significantly challenging. This paper presents Neural-Rendezvous, a deep learning-based guidance and control framework for encountering fast-moving objects, including ISOs, robustly, accurately, and autonomously in real time. It uses pointwise minimum norm tracking control on top of a guidance policy modeled by a spectrally-normalized deep neural network, where its hyperparameters are tuned with a loss function directly penalizing the MPC state trajectory tracking error. We show that Neural-Rendezvous provides a high probability exponential bound on the expected spacecraft delivery error, the proof of which leverages stochastic incremental stability analysis. In particular, it is used to
    

