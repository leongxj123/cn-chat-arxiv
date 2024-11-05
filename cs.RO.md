# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Visual Whole-Body Control for Legged Loco-Manipulation](https://arxiv.org/abs/2403.16967) | 这项研究提出了一种利用视觉全身控制的框架，使腿式机器人能够同时控制腿部和手臂，以扩展操作能力，并通过仿真训练和Sim2Real转移实现了在捡起不同物体方面取得显著改进。 |
| [^2] | [Mixed-Strategy Nash Equilibrium for Crowd Navigation](https://arxiv.org/abs/2403.01537) | 通过简单的迭代贝叶斯更新方案和基于数据驱动的框架，我们证明了混合策略纳什均衡模型为人群导航提供了实时且可扩展的决策制定方法。 |
| [^3] | [Towards Inferring Users' Impressions of Robot Performance in Navigation Scenarios.](http://arxiv.org/abs/2310.11590) | 本研究拟通过非语言行为提示和机器学习技术预测人们对机器人行为印象，并提供了一个数据集和分析结果，发现在导航场景中，空间特征是最关键的信息。 |
| [^4] | [Zero-Shot Wireless Indoor Navigation through Physics-Informed Reinforcement Learning.](http://arxiv.org/abs/2306.06766) | 该论文提出了一种基于物理信息强化学习的零样本无线室内导航方法，通过样本高效学习和零样本泛化来提高导航效率。 |
| [^5] | [Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections.](http://arxiv.org/abs/2301.05294) | 本研究提出了一种去中心化的多智能体强化学习方法，用于控制和协调混合交通，特别是人驾驶车辆和机器人车辆在实际复杂交叉口的应用。实验结果表明，使用5%的机器人车辆可以有效防止交叉口内的拥堵形成。 |
| [^6] | [Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects.](http://arxiv.org/abs/2208.04883) | 本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。 |

# 详细

[^1]: 用于腿式定点机器人运动操作的视觉全身控制

    Visual Whole-Body Control for Legged Loco-Manipulation

    [https://arxiv.org/abs/2403.16967](https://arxiv.org/abs/2403.16967)

    这项研究提出了一种利用视觉全身控制的框架，使腿式机器人能够同时控制腿部和手臂，以扩展操作能力，并通过仿真训练和Sim2Real转移实现了在捡起不同物体方面取得显著改进。

    

    我们研究了使用配备手臂的腿式机器人进行移动操作的问题，即腿式定点操作。尽管机器人的腿通常用于移动，但通过进行全身控制，可以扩大其操作能力。也就是说，机器人可以同时控制腿部和手臂，以扩展其工作空间。我们提出了一个能够使用视觉观测自主进行全身控制的框架。我们的方法称为\ourFull~(\our)，由一个低级策略和一个高级策略组成。低级策略使用所有自由度来跟踪末端执行器的位置，高级策略根据视觉输入提出末端执行器位置。我们在仿真中训练了两个级别的策略，并进行了从Sim到实物的转移以进行实际机器人部署。我们进行了大量实验证明，在不同配置下（高度、）捡起不同物体方面，相对基线方法取得了显著改进。

    arXiv:2403.16967v1 Announce Type: cross  Abstract: We study the problem of mobile manipulation using legged robots equipped with an arm, namely legged loco-manipulation. The robot legs, while usually utilized for mobility, offer an opportunity to amplify the manipulation capabilities by conducting whole-body control. That is, the robot can control the legs and the arm at the same time to extend its workspace. We propose a framework that can conduct the whole-body control autonomously with visual observations. Our approach, namely \ourFull~(\our), is composed of a low-level policy using all degrees of freedom to track the end-effector manipulator position and a high-level policy proposing the end-effector position based on visual inputs. We train both levels of policies in simulation and perform Sim2Real transfer for real robot deployment. We perform extensive experiments and show significant improvements over baselines in picking up diverse objects in different configurations (heights,
    
[^2]: 混合策略纳什均衡用于人群导航

    Mixed-Strategy Nash Equilibrium for Crowd Navigation

    [https://arxiv.org/abs/2403.01537](https://arxiv.org/abs/2403.01537)

    通过简单的迭代贝叶斯更新方案和基于数据驱动的框架，我们证明了混合策略纳什均衡模型为人群导航提供了实时且可扩展的决策制定方法。

    

    我们解决了针对人群导航找到混合策略纳什均衡的问题。混合策略纳什均衡为机器人提供了一个严谨的模型，使其能够预测人群中不确定但合作的人类行为，但计算成本通常太高，无法进行可扩展和实时的决策制定。在这里，我们证明了一个简单的迭代贝叶斯更新方案收敛于混合策略社交导航游戏的纳什均衡。此外，我们提出了一个基于数据驱动的框架，通过将代理策略初始化为从人类数据集学习的高斯过程，来构建该游戏。基于所提出的混合策略纳什均衡模型，我们开发了一个基于采样的人群导航框架，可以集成到现有导航方法中，并可在笔记本电脑 CPU 上实时运行。我们通过模拟环境和真实世界的非结构化环境中人类数据集对我们的框架进行了评估。

    arXiv:2403.01537v1 Announce Type: cross  Abstract: We address the problem of finding mixed-strategy Nash equilibrium for crowd navigation. Mixed-strategy Nash equilibrium provides a rigorous model for the robot to anticipate uncertain yet cooperative human behavior in crowds, but the computation cost is often too high for scalable and real-time decision-making. Here we prove that a simple iterative Bayesian updating scheme converges to the Nash equilibrium of a mixed-strategy social navigation game. Furthermore, we propose a data-driven framework to construct the game by initializing agent strategies as Gaussian processes learned from human datasets. Based on the proposed mixed-strategy Nash equilibrium model, we develop a sampling-based crowd navigation framework that can be integrated into existing navigation methods and runs in real-time on a laptop CPU. We evaluate our framework in both simulated environments and real-world human datasets in unstructured environments. Our framework
    
[^3]: 探索在导航场景下推断用户对机器人性能的印象

    Towards Inferring Users' Impressions of Robot Performance in Navigation Scenarios. (arXiv:2310.11590v1 [cs.RO])

    [http://arxiv.org/abs/2310.11590](http://arxiv.org/abs/2310.11590)

    本研究拟通过非语言行为提示和机器学习技术预测人们对机器人行为印象，并提供了一个数据集和分析结果，发现在导航场景中，空间特征是最关键的信息。

    

    人们对机器人性能的印象通常通过调查问卷来衡量。作为一种更可扩展且成本效益更高的替代方案，我们研究了使用非语言行为提示和机器学习技术预测人们对机器人行为印象的可能性。为此，我们首先提供了SEAN TOGETHER数据集，该数据集包括在虚拟现实模拟中人与移动机器人相互作用的观察结果，以及用户对机器人性能的5点量表评价。其次，我们对人类和监督学习技术如何基于不同的观察类型（例如面部、空间和地图特征）来预测感知到的机器人性能进行了分析。我们的结果表明，仅仅面部表情就能提供关于人们对机器人性能印象的有用信息；但在我们测试的导航场景中，空间特征是这种推断任务最关键的信息。

    Human impressions of robot performance are often measured through surveys. As a more scalable and cost-effective alternative, we study the possibility of predicting people's impressions of robot behavior using non-verbal behavioral cues and machine learning techniques. To this end, we first contribute the SEAN TOGETHER Dataset consisting of observations of an interaction between a person and a mobile robot in a Virtual Reality simulation, together with impressions of robot performance provided by users on a 5-point scale. Second, we contribute analyses of how well humans and supervised learning techniques can predict perceived robot performance based on different combinations of observation types (e.g., facial, spatial, and map features). Our results show that facial expressions alone provide useful information about human impressions of robot performance; but in the navigation scenarios we tested, spatial features are the most critical piece of information for this inference task. Als
    
[^4]: 零样本无线室内导航通过物理信息强化学习

    Zero-Shot Wireless Indoor Navigation through Physics-Informed Reinforcement Learning. (arXiv:2306.06766v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2306.06766](http://arxiv.org/abs/2306.06766)

    该论文提出了一种基于物理信息强化学习的零样本无线室内导航方法，通过样本高效学习和零样本泛化来提高导航效率。

    

    针对室内机器人导航利用无线信号的不断关注，该论文提出了一种基于物理信息强化学习（PIRL）的方法，以实现高效的样本学习和零样本泛化。相对于基于启发式方法的传统方法，基于射频传播的方法直观且能够适应简单的场景，但在复杂环境下难以导航。而基于端到端深度强化学习（RL）的方法可以通过使用先进的计算机技术来探索整个状态空间，在面对复杂的无线环境时表现出令人惊讶的性能。然而，这种方法需要大量的训练样本，而且得到的策略在未进行微调的情况下无法在训练阶段未见过的新场景中有效导航。

    The growing focus on indoor robot navigation utilizing wireless signals has stemmed from the capability of these signals to capture high-resolution angular and temporal measurements. Prior heuristic-based methods, based on radio frequency propagation, are intuitive and generalizable across simple scenarios, yet fail to navigate in complex environments. On the other hand, end-to-end (e2e) deep reinforcement learning (RL), powered by advanced computing machinery, can explore the entire state space, delivering surprising performance when facing complex wireless environments. However, the price to pay is the astronomical amount of training samples, and the resulting policy, without fine-tuning (zero-shot), is unable to navigate efficiently in new scenarios unseen in the training phase. To equip the navigation agent with sample-efficient learning and {zero-shot} generalization, this work proposes a novel physics-informed RL (PIRL) where a distance-to-target-based cost (standard in e2e) is a
    
[^5]: 通过机器人车辆在复杂和无信号的交叉口中学习控制和协调混合交通

    Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections. (arXiv:2301.05294v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.05294](http://arxiv.org/abs/2301.05294)

    本研究提出了一种去中心化的多智能体强化学习方法，用于控制和协调混合交通，特别是人驾驶车辆和机器人车辆在实际复杂交叉口的应用。实验结果表明，使用5%的机器人车辆可以有效防止交叉口内的拥堵形成。

    

    交叉口是现代大都市交通中必不可少的道路基础设施。然而，由于交通事故或缺乏交通协调机制（如交通信号灯），它们也可能成为交通流的瓶颈。最近，提出了各种超越传统控制方法的控制和协调机制，以提高交叉口交通的效率。在这些方法中，控制可预见的包含人驾驶车辆（HVs）和机器人车辆（RVs）的混合交通已经出现。在本项目中，我们提出了一种去中心化的多智能体强化学习方法，用于实际复杂交叉口的混合交通的控制和协调，这是一个以前未被探索过的主题。我们进行了全面的实验，以展示我们方法的有效性。特别是，我们展示了在实际交通条件下，使用5%的RVs，我们可以防止复杂交叉口内的拥堵形成。

    Intersections are essential road infrastructures for traffic in modern metropolises. However, they can also be the bottleneck of traffic flows as a result of traffic incidents or the absence of traffic coordination mechanisms such as traffic lights. Recently, various control and coordination mechanisms that are beyond traditional control methods have been proposed to improve the efficiency of intersection traffic. Amongst these methods, the control of foreseeable mixed traffic that consists of human-driven vehicles (HVs) and robot vehicles (RVs) has emerged. In this project, we propose a decentralized multi-agent reinforcement learning approach for the control and coordination of mixed traffic at real-world, complex intersections--a topic that has not been previously explored. Comprehensive experiments are conducted to show the effectiveness of our approach. In particular, we show that using 5% RVs, we can prevent congestion formation inside a complex intersection under the actual traf
    
[^6]: 神经会合：面向星际物体的可靠导航和控制的证明

    Neural-Rendezvous: Provably Robust Guidance and Control to Encounter Interstellar Objects. (arXiv:2208.04883v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2208.04883](http://arxiv.org/abs/2208.04883)

    本文提出了神经会合，一种深度学习导航和控制框架，用于可靠、准确和自主地遭遇快速移动的星际物体。它通过点最小范数追踪控制和谱归一化深度神经网络引导策略来提供高概率指数上界的飞行器交付误差。

    

    星际物体（ISOs）很可能是不可替代的原始材料，在理解系外行星星系方面具有重要价值。然而，由于其运行轨道难以约束，通常具有较高的倾角和相对速度，使用传统的人在环路方法探索ISOs具有相当大的挑战性。本文提出了一种名为神经会合的深度学习导航和控制框架，用于在实时中以可靠、准确和自主的方式遭遇快速移动的物体，包括ISOs。它在基于谱归一化的深度神经网络的引导策略之上使用点最小范数追踪控制，其中参数通过直接惩罚MPC状态轨迹跟踪误差的损失函数进行调优。我们展示了神经会合在预期的飞行器交付误差上提供了高概率指数上界，其证明利用了随机递增稳定性分析。

    Interstellar objects (ISOs) are likely representatives of primitive materials invaluable in understanding exoplanetary star systems. Due to their poorly constrained orbits with generally high inclinations and relative velocities, however, exploring ISOs with conventional human-in-the-loop approaches is significantly challenging. This paper presents Neural-Rendezvous, a deep learning-based guidance and control framework for encountering fast-moving objects, including ISOs, robustly, accurately, and autonomously in real time. It uses pointwise minimum norm tracking control on top of a guidance policy modeled by a spectrally-normalized deep neural network, where its hyperparameters are tuned with a loss function directly penalizing the MPC state trajectory tracking error. We show that Neural-Rendezvous provides a high probability exponential bound on the expected spacecraft delivery error, the proof of which leverages stochastic incremental stability analysis. In particular, it is used to
    

