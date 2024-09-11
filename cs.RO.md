# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ViSaRL: Visual Reinforcement Learning Guided by Human Saliency](https://arxiv.org/abs/2403.10940) | ViSaRL提出了Visual Saliency-Guided Reinforcement Learning（受视觉显著性引导的强化学习）方法，通过学习视觉表示来显著提高RL代理在不同任务上的成功率、样本效率和泛化性能。 |
| [^2] | [Greedy Perspectives: Multi-Drone View Planning for Collaborative Coverage in Cluttered Environments.](http://arxiv.org/abs/2310.10863) | 本研究研究了在杂乱环境中协调无人机团队拍摄复杂人群的多无人机多演员视角规划问题，并开发了一个具有遮挡感知目标的视角规划器进行性能比较。 |
| [^3] | [Finite element inspired networks: Learning physically-plausible deformable object dynamics from partial observations.](http://arxiv.org/abs/2307.07975) | 该论文提出了一种基于有限元的网络模型，通过动力学网络和物理感知编码器，从部分观察中学习可变形物体的动力学，并通过正运动学解码器进行预测，实现了具有物理解释性的模型。 |
| [^4] | [System Neural Diversity: Measuring Behavioral Heterogeneity in Multi-Agent Learning.](http://arxiv.org/abs/2305.02128) | 本文介绍了一种名为“系统神经多样性”的方法，用于度量具有随机策略的多智能体系统的行为异质性，探讨了多样性对集体弹性和性能的影响。 |

# 详细

[^1]: ViSaRL：受人类显著性引导的视觉强化学习

    ViSaRL: Visual Reinforcement Learning Guided by Human Saliency

    [https://arxiv.org/abs/2403.10940](https://arxiv.org/abs/2403.10940)

    ViSaRL提出了Visual Saliency-Guided Reinforcement Learning（受视觉显著性引导的强化学习）方法，通过学习视觉表示来显著提高RL代理在不同任务上的成功率、样本效率和泛化性能。

    

    使用强化学习（RL）从高维像素输入培训机器人执行复杂控制任务在样本效率上是低效的，因为图像观察主要由与任务无关的信息组成。相比之下，人类能够在视觉上关注与任务相关的对象和区域。基于这一观察，我们引入了受视觉显著性引导的强化学习（ViSaRL）。使用ViSaRL学习视觉表示显着提高了RL代理在不同任务上，包括DeepMind控制基准、仿真中的机器人操作和真实机器人上的成功率、样本效率和泛化性能。我们提出了将显著性整合到基于CNN和Transformer的编码器中的方法。我们展示使用ViSaRL学习的视觉表示对各种视觉扰动，包括感知噪声和场景变化，都具有鲁棒性。ViSaRL在真实环境中成功率几乎翻了一番。

    arXiv:2403.10940v1 Announce Type: cross  Abstract: Training robots to perform complex control tasks from high-dimensional pixel input using reinforcement learning (RL) is sample-inefficient, because image observations are comprised primarily of task-irrelevant information. By contrast, humans are able to visually attend to task-relevant objects and areas. Based on this insight, we introduce Visual Saliency-Guided Reinforcement Learning (ViSaRL). Using ViSaRL to learn visual representations significantly improves the success rate, sample efficiency, and generalization of an RL agent on diverse tasks including DeepMind Control benchmark, robot manipulation in simulation and on a real robot. We present approaches for incorporating saliency into both CNN and Transformer-based encoders. We show that visual representations learned using ViSaRL are robust to various sources of visual perturbations including perceptual noise and scene variations. ViSaRL nearly doubles success rate on the real-
    
[^2]: 贪心视角：多无人机视野规划在杂乱环境中的协同覆盖

    Greedy Perspectives: Multi-Drone View Planning for Collaborative Coverage in Cluttered Environments. (arXiv:2310.10863v1 [cs.RO])

    [http://arxiv.org/abs/2310.10863](http://arxiv.org/abs/2310.10863)

    本研究研究了在杂乱环境中协调无人机团队拍摄复杂人群的多无人机多演员视角规划问题，并开发了一个具有遮挡感知目标的视角规划器进行性能比较。

    

    无人机团队的部署可以在复杂环境中拍摄动态人群（演员）的大规模影像，用于团队运动和电影制作等新应用领域。为了实现该目标，可以使用通过顺序贪心规划进行子模最大化的方法，以便在无人机团队之间进行摄像机视野的可扩展优化，但在杂乱环境中协同效果面临挑战。障碍物可能产生遮挡并增加无人机碰撞的几率，这可能违反近似最优性的要求。为了在稠密环境中协调无人机团队拍摄人群，需要一种更通用的视角规划方法。我们通过开发一个具有遮挡感知目标的多无人机多演员视角规划器，并与贪心形成规划器进行比较，探讨遮挡和碰撞对拍摄应用性能的影响。为了评估性能，

    Deployment of teams of aerial robots could enable large-scale filming of dynamic groups of people (actors) in complex environments for novel applications in areas such as team sports and cinematography. Toward this end, methods for submodular maximization via sequential greedy planning can be used for scalable optimization of camera views across teams of robots but face challenges with efficient coordination in cluttered environments. Obstacles can produce occlusions and increase chances of inter-robot collision which can violate requirements for near-optimality guarantees. To coordinate teams of aerial robots in filming groups of people in dense environments, a more general view-planning approach is required. We explore how collision and occlusion impact performance in filming applications through the development of a multi-robot multi-actor view planner with an occlusion-aware objective for filming groups of people and compare with a greedy formation planner. To evaluate performance,
    
[^3]: 基于有限元的网络: 从部分观察中学习合理的可变形物体动力学

    Finite element inspired networks: Learning physically-plausible deformable object dynamics from partial observations. (arXiv:2307.07975v1 [cs.RO])

    [http://arxiv.org/abs/2307.07975](http://arxiv.org/abs/2307.07975)

    该论文提出了一种基于有限元的网络模型，通过动力学网络和物理感知编码器，从部分观察中学习可变形物体的动力学，并通过正运动学解码器进行预测，实现了具有物理解释性的模型。

    

    精确模拟可变形线性物体（DLO）的动力学在需要一个可以被人解读和数据高效的模型并能够提供快速预测的任务中是具有挑战性的。为了得到这样的模型，我们借鉴了刚性有限元方法（R-FEM），将DLO建模为一系列刚体链，其内部状态通过动力学网络以时间展开。由于该状态不能直接观察到，动力学网络与一个物理感知的编码器共同训练，将观察到的运动变量映射到刚体链的状态。为了促使状态获得物理上有意义的表示，我们利用底层R-FEM模型的正运动学（FK）作为解码器。我们在一个机器人实验中证明，这种被称为“有限元启发网络”的架构是一个易于处理但功能强大的DLO动力学模型，可以从部分观察中得出具有物理解释性的预测。

    The accurate simulation of deformable linear object (DLO) dynamics is challenging if the task at hand requires a human-interpretable and data-efficient model that also yields fast predictions. To arrive at such model, we draw inspiration from the rigid finite element method (R-FEM) and model a DLO as a serial chain of rigid bodies whose internal state is unrolled through time by a dynamics network. As this state is not observed directly, the dynamics network is trained jointly with a physics-informed encoder mapping observed motion variables to the body chain's state. To encourage that the state acquires a physically meaningful representation, we leverage the forward kinematics (FK) of the underlying R-FEM model as a decoder. We demonstrate in a robot experiment that this architecture - being termed "Finite element inspired network" - forms an easy to handle, yet capable DLO dynamics model yielding physically interpretable predictions from partial observations.  The project code is ava
    
[^4]: 系统神经多样性：在多智能体学习中度量行为异质性

    System Neural Diversity: Measuring Behavioral Heterogeneity in Multi-Agent Learning. (arXiv:2305.02128v1 [cs.MA])

    [http://arxiv.org/abs/2305.02128](http://arxiv.org/abs/2305.02128)

    本文介绍了一种名为“系统神经多样性”的方法，用于度量具有随机策略的多智能体系统的行为异质性，探讨了多样性对集体弹性和性能的影响。

    

    进化科学提供了多样性具有韧性的证据。然而，传统的多智能体强化学习技术通常强制要求同质性以增加训练样本的效率。当学习代理系统不受同质策略的限制时，个体代理可能会发展出不同的行为，从而产生有利于系统的新兴互补性。尽管如此，缺乏衡量学习代理系统中行为多样性的工具意味着我们无法深入了解多样性对集体弹性和性能的影响。在本文中，我们介绍了系统神经多样性（SND）：一种用于具有随机策略的多智能体系统的行为异质性度量方法，探讨并证明了其理论性质，并将其与跨学科领域中使用的最新行为多样性指标进行了比较。

    Evolutionary science provides evidence that diversity confers resilience. Yet, traditional multi-agent reinforcement learning techniques commonly enforce homogeneity to increase training sample efficiency. When a system of learning agents is not constrained to homogeneous policies, individual agents may develop diverse behaviors, resulting in emergent complementarity that benefits the system. Despite this feat, there is a surprising lack of tools that measure behavioral diversity in systems of learning agents. Such techniques would pave the way towards understanding the impact of diversity in collective resilience and performance. In this paper, we introduce System Neural Diversity (SND): a measure of behavioral heterogeneity for multi-agent systems where agents have stochastic policies. %over a continuous state space. We discuss and prove its theoretical properties, and compare it with alternate, state-of-the-art behavioral diversity metrics used in cross-disciplinary domains. Through
    

