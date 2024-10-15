# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Quadruped Locomotion Using Differentiable Simulation](https://arxiv.org/abs/2403.14864) | 本文提出了一种新的可微分仿真框架，通过将复杂的全身仿真解耦为两个单独的连续域，并与更精确的模型对齐，来克服四足动作中的不连续性挑战。 |
| [^2] | [A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2403.10996) | 提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题 |
| [^3] | [Twisting Lids Off with Two Hands](https://arxiv.org/abs/2403.02338) | 深度强化学习结合仿真到真实世界的转移为解决物体操纵问题提供了有力支持 |
| [^4] | [S.T.A.R.-Track: Latent Motion Models for End-to-End 3D Object Tracking with Adaptive Spatio-Temporal Appearance Representations.](http://arxiv.org/abs/2306.17602) | 本文提出了S.T.A.R.-Track，一个采用物体为中心的Transformer框架，用于端到端3D物体跟踪。通过新颖的潜在运动模型和学习型跟踪嵌入，该框架能够准确建模物体的几何运动和变化，并在nuScenes数据集上取得了优秀的性能。 |
| [^5] | [DiMSam: Diffusion Models as Samplers for Task and Motion Planning under Partial Observability.](http://arxiv.org/abs/2306.13196) | 本文提出了一种使用扩散模型作为采样器的任务和动作规划方法，在部分可观测下能够实现长周期受约束的操作计划。 |
| [^6] | [Privileged Knowledge Distillation for Sim-to-Real Policy Generalization.](http://arxiv.org/abs/2305.18464) | 论文提出了一种新的单阶段特权知识蒸馏方法（HIB），通过捕捉历史轨迹的特权知识表示来学习，缩小模拟和真实之间的差距。 |

# 详细

[^1]: 使用可微分仿真学习四足动作

    Learning Quadruped Locomotion Using Differentiable Simulation

    [https://arxiv.org/abs/2403.14864](https://arxiv.org/abs/2403.14864)

    本文提出了一种新的可微分仿真框架，通过将复杂的全身仿真解耦为两个单独的连续域，并与更精确的模型对齐，来克服四足动作中的不连续性挑战。

    

    最近大部分机器人运动控制的进展都是由无模型强化学习驱动的，本文探讨了可微分仿真的潜力。可微分仿真通过使用机器人模型计算低变异一阶梯度，承诺了更快的收敛速度和更稳定的训练，但到目前为止，其在四足机器人控制方面的应用仍然有限。可微分仿真面临的主要挑战在于由于接触丰富环境（如四足动作）中的不连续性，导致机器人任务的复杂优化景观。本文提出了一个新的可微分仿真框架以克服这些挑战。关键想法包括将可能由于接触而出现不连续性的复杂全身仿真解耦为两个单独的连续域。随后，我们将简化模型产生的机器人状态与更精确的不可微分模型对齐。

    arXiv:2403.14864v1 Announce Type: cross  Abstract: While most recent advancements in legged robot control have been driven by model-free reinforcement learning, we explore the potential of differentiable simulation. Differentiable simulation promises faster convergence and more stable training by computing low-variant first-order gradients using the robot model, but so far, its use for legged robot control has remained limited to simulation. The main challenge with differentiable simulation lies in the complex optimization landscape of robotic tasks due to discontinuities in contact-rich environments, e.g., quadruped locomotion. This work proposes a new, differentiable simulation framework to overcome these challenges. The key idea involves decoupling the complex whole-body simulation, which may exhibit discontinuities due to contact, into two separate continuous domains. Subsequently, we align the robot state resulting from the simplified model with a more precise, non-differentiable 
    
[^2]: 一个可扩展且可并行化的数字孪生框架，用于多智能体强化学习系统可持续Sim2Real转换

    A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2403.10996](https://arxiv.org/abs/2403.10996)

    提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题

    

    本工作提出了一个可持续的多智能体深度强化学习框架，能够选择性地按需扩展并行化训练工作负载，并利用最少的硬件资源将训练好的策略从模拟环境转移到现实世界。我们引入了AutoDRIVE生态系统作为一个启动数字孪生框架，用于训练、部署和转移合作和竞争的多智能体强化学习策略从模拟环境到现实世界。具体来说，我们首先探究了4台合作车辆(Nigel)在单智能体和多智能体学习环境中共享有限状态信息的交叉遍历问题，采用了一种通用策略方法。然后，我们使用个体策略方法研究了2辆车(F1TENTH)的对抗性自主赛车问题。在任何一组实验中，我们采用了去中心化学习架构，这允许对策略进行有力的训练和测试。

    arXiv:2403.10996v1 Announce Type: cross  Abstract: This work presents a sustainable multi-agent deep reinforcement learning framework capable of selectively scaling parallelized training workloads on-demand, and transferring the trained policies from simulation to reality using minimal hardware resources. We introduce AutoDRIVE Ecosystem as an enabling digital twin framework to train, deploy, and transfer cooperative as well as competitive multi-agent reinforcement learning policies from simulation to reality. Particularly, we first investigate an intersection traversal problem of 4 cooperative vehicles (Nigel) that share limited state information in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial autonomous racing problem of 2 vehicles (F1TENTH) using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted, which allowed robust training and testing of the policies 
    
[^3]: 用双手扭开盖子

    Twisting Lids Off with Two Hands

    [https://arxiv.org/abs/2403.02338](https://arxiv.org/abs/2403.02338)

    深度强化学习结合仿真到真实世界的转移为解决物体操纵问题提供了有力支持

    

    用两只多指手臂操纵物体一直是机器人领域的一项长期挑战，原因在于许多操纵任务的丰富接触性质以及协调高维度双手系统固有的复杂性。在这项工作中，我们考虑了使用两只手扭开各种瓶子盖的问题，并展示出使用深度强化学习在仿真中训练的策略可以有效地转移到现实世界。通过对物理建模、实时感知和奖励设计的新工程见解，该策略展示了一般化能力，能够贯穿各种看不见的物体，展示出动态和灵巧的行为。我们的发现证明了深度强化学习结合仿真到真实世界的转移仍然是解决前所未有复杂问题的操纵问题的一个有前途的方法。

    arXiv:2403.02338v1 Announce Type: cross  Abstract: Manipulating objects with two multi-fingered hands has been a long-standing challenge in robotics, attributed to the contact-rich nature of many manipulation tasks and the complexity inherent in coordinating a high-dimensional bimanual system. In this work, we consider the problem of twisting lids of various bottle-like objects with two hands, and demonstrate that policies trained in simulation using deep reinforcement learning can be effectively transferred to the real world. With novel engineering insights into physical modeling, real-time perception, and reward design, the policy demonstrates generalization capabilities across a diverse set of unseen objects, showcasing dynamic and dexterous behaviors. Our findings serve as compelling evidence that deep reinforcement learning combined with sim-to-real transfer remains a promising approach for addressing manipulation problems of unprecedented complexity.
    
[^4]: S.T.A.R.-Track：自适应时空外貌表示的端到端3D物体跟踪的潜在运动模型

    S.T.A.R.-Track: Latent Motion Models for End-to-End 3D Object Tracking with Adaptive Spatio-Temporal Appearance Representations. (arXiv:2306.17602v1 [cs.CV])

    [http://arxiv.org/abs/2306.17602](http://arxiv.org/abs/2306.17602)

    本文提出了S.T.A.R.-Track，一个采用物体为中心的Transformer框架，用于端到端3D物体跟踪。通过新颖的潜在运动模型和学习型跟踪嵌入，该框架能够准确建模物体的几何运动和变化，并在nuScenes数据集上取得了优秀的性能。

    

    本文基于跟踪-注意力模式，引入了一个以物体为中心的基于Transformer的3D跟踪框架。传统的基于模型的跟踪方法通过几何运动模型融合帧之间的物体和自运动的几何效应。受此启发，我们提出了S.T.A.R.-Track，使用一种新颖的潜在运动模型来调整对象查询，以在潜在空间中直接考虑视角和光照条件的变化，同时明确建模几何运动。结合一种新颖的可学习的跟踪嵌入，有助于建模轨迹的存在概率，这导致了一个通用的跟踪框架，可以与任何基于查询的检测器集成。在nuScenes基准测试上进行了大量实验，证明了我们方法的优势，展示了基于DETR3D的跟踪器的最先进性能，同时大大减少了轨迹的身份转换次数。

    Following the tracking-by-attention paradigm, this paper introduces an object-centric, transformer-based framework for tracking in 3D. Traditional model-based tracking approaches incorporate the geometric effect of object- and ego motion between frames with a geometric motion model. Inspired by this, we propose S.T.A.R.-Track, which uses a novel latent motion model (LMM) to additionally adjust object queries to account for changes in viewing direction and lighting conditions directly in the latent space, while still modeling the geometric motion explicitly. Combined with a novel learnable track embedding that aids in modeling the existence probability of tracks, this results in a generic tracking framework that can be integrated with any query-based detector. Extensive experiments on the nuScenes benchmark demonstrate the benefits of our approach, showing state-of-the-art performance for DETR3D-based trackers while drastically reducing the number of identity switches of tracks at the s
    
[^5]: DiMSam:扩散模型作为部分可观测任务与动作规划中的采样器。

    DiMSam: Diffusion Models as Samplers for Task and Motion Planning under Partial Observability. (arXiv:2306.13196v1 [cs.RO])

    [http://arxiv.org/abs/2306.13196](http://arxiv.org/abs/2306.13196)

    本文提出了一种使用扩散模型作为采样器的任务和动作规划方法，在部分可观测下能够实现长周期受约束的操作计划。

    

    任务和动作规划（TAMP）方法非常有效地计划长周期自主机器人操作。但是，由于它们需要一个规划模型，因此在环境和其动态不完全了解的领域中应用它们可能非常困难。我们提出通过利用深度生成建模，特别是扩散模型来克服这些限制，学习捕获规划模型中难以设计的约束和采样器。这些学习采样器在TAMP求解器中组合和合并，以联合找到满足规划中约束的行动参数值。为了便于对环境中未知对象进行预测，我们将这些采样器定义为学习的低维潜变量嵌入的可变对象状态。我们在关节式物体操作领域评估了我们的方法，并展示了经典TAMP、生成学习和潜在嵌入的组合如何使得在部分可观测下进行长周期受约束的操作计划。

    Task and Motion Planning (TAMP) approaches are effective at planning long-horizon autonomous robot manipulation. However, because they require a planning model, it can be difficult to apply them to domains where the environment and its dynamics are not fully known. We propose to overcome these limitations by leveraging deep generative modeling, specifically diffusion models, to learn constraints and samplers that capture these difficult-to-engineer aspects of the planning model. These learned samplers are composed and combined within a TAMP solver in order to find action parameter values jointly that satisfy the constraints along a plan. To tractably make predictions for unseen objects in the environment, we define these samplers on low-dimensional learned latent embeddings of changing object state. We evaluate our approach in an articulated object manipulation domain and show how the combination of classical TAMP, generative learning, and latent embeddings enables long-horizon constra
    
[^6]: 特权知识蒸馏用于 Sim-to-Real 策略泛化

    Privileged Knowledge Distillation for Sim-to-Real Policy Generalization. (arXiv:2305.18464v1 [cs.LG])

    [http://arxiv.org/abs/2305.18464](http://arxiv.org/abs/2305.18464)

    论文提出了一种新的单阶段特权知识蒸馏方法（HIB），通过捕捉历史轨迹的特权知识表示来学习，缩小模拟和真实之间的差距。

    

    强化学习最近在机器人控制方面取得了显著的成功。但是，多数强化学习方法在模拟环境中运行，那里的特权知识（例如动力学，环境，地形）是轻松获取的。相反，在真实场景中，机器人代理通常仅依赖于本地状态（例如机器人关节的本体感反馈）来选择动作，导致显著的模拟到真实的差距。现有方法通过逐渐减少对特权知识的依赖或执行两阶段策略模仿来解决这个差距。但我们认为，这些方法在充分利用特权知识的能力方面存在局限性，导致性能次优。本文提出了一种称为历史信息瓶颈（HIB）的新型单阶段特权知识蒸馏方法，以缩小模拟到真实的差距。具体而言，HIB通过捕捉历史轨迹的特权知识表示来学习。

    Reinforcement Learning (RL) has recently achieved remarkable success in robotic control. However, most RL methods operate in simulated environments where privileged knowledge (e.g., dynamics, surroundings, terrains) is readily available. Conversely, in real-world scenarios, robot agents usually rely solely on local states (e.g., proprioceptive feedback of robot joints) to select actions, leading to a significant sim-to-real gap. Existing methods address this gap by either gradually reducing the reliance on privileged knowledge or performing a two-stage policy imitation. However, we argue that these methods are limited in their ability to fully leverage the privileged knowledge, resulting in suboptimal performance. In this paper, we propose a novel single-stage privileged knowledge distillation method called the Historical Information Bottleneck (HIB) to narrow the sim-to-real gap. In particular, HIB learns a privileged knowledge representation from historical trajectories by capturing 
    

