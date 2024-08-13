# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Elastic Time Steps](https://arxiv.org/abs/2402.14961) | SEAC是一种弹性时间步长的离策略演员-评论家算法，通过可变持续时间的时间步长，使代理能够根据情况改变控制频率，在模拟环境中表现优异。 |
| [^2] | [KIX: A Metacognitive Generalization Framework](https://arxiv.org/abs/2402.05346) | 人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。 |
| [^3] | ["Task Success" is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors](https://arxiv.org/abs/2402.04210) | 本文探究了将视频-语言模型作为行为批评者用于捕捉不良代理行为的可行性，以解决大规模生成模型忽视任务约束和用户偏好的问题。 |
| [^4] | [Toward a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning](https://arxiv.org/abs/2311.17693) | 利用模拟图像引导的以外科医生为中心的自主机器人学徒系统，通过强化学习和模仿学习代理在眼科白内障手术中适应外科医生的技能水平和外科手术偏好。 |
| [^5] | [Constant-time Motion Planning with Anytime Refinement for Manipulation.](http://arxiv.org/abs/2311.00837) | 我们提出了一种任意时间提炼方法，结合恒定时间动作规划器（CTMP）在机器人操作中改善解决方案。这种方法利用机器人系统拥有的多余计划时间来完善运动规划的结果。 |
| [^6] | [CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration.](http://arxiv.org/abs/2309.14660) | CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。 |

# 详细

[^1]: 弹性时间步长的强化学习

    Reinforcement Learning with Elastic Time Steps

    [https://arxiv.org/abs/2402.14961](https://arxiv.org/abs/2402.14961)

    SEAC是一种弹性时间步长的离策略演员-评论家算法，通过可变持续时间的时间步长，使代理能够根据情况改变控制频率，在模拟环境中表现优异。

    

    传统的强化学习（RL）算法通常应用于机器人学习以以固定控制频率执行动作的控制器。鉴于RL算法的离散性质，它们对控制频率的选择的影响视而不见：找到正确的控制频率可能很困难，错误往往会导致过度使用计算资源甚至导致无法收敛。我们提出了软弹性演员-评论家（SEAC）, 一种新颖的离策略演员-评论家算法来解决这个问题。SEAC实现了弹性时间步长，即具有已知变化持续时间的时间步长，允许代理根据情况改变其控制频率。在实践中，SEAC仅在必要时应用控制，最小化计算资源和数据使用。我们在模拟环境中评估了SEAC在牛顿运动学迷宫导航任务和三维赛车视频游戏Trackmania中的能力。SEAC在表现上优于SAC基线。

    arXiv:2402.14961v1 Announce Type: cross  Abstract: Traditional Reinforcement Learning (RL) algorithms are usually applied in robotics to learn controllers that act with a fixed control rate. Given the discrete nature of RL algorithms, they are oblivious to the effects of the choice of control rate: finding the correct control rate can be difficult and mistakes often result in excessive use of computing resources or even lack of convergence.   We propose Soft Elastic Actor-Critic (SEAC), a novel off-policy actor-critic algorithm to address this issue. SEAC implements elastic time steps, time steps with a known, variable duration, which allow the agent to change its control frequency to adapt to the situation. In practice, SEAC applies control only when necessary, minimizing computational resources and data usage.   We evaluate SEAC's capabilities in simulation in a Newtonian kinematics maze navigation task and on a 3D racing video game, Trackmania. SEAC outperforms the SAC baseline in t
    
[^2]: KIX: 一种元认知泛化框架

    KIX: A Metacognitive Generalization Framework

    [https://arxiv.org/abs/2402.05346](https://arxiv.org/abs/2402.05346)

    人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。

    

    人类和其他动物能够灵活解决各种任务，并且能够通过重复使用和应用长期积累的高级知识来适应新颖情境，这表现了一种泛化智能行为。但是人工智能代理更多地是专家，缺乏这种通用行为。人工智能代理需要理解和利用关键的结构化知识表示。我们提出了一种元认知泛化框架，称为Knowledge-Interaction-eXecution (KIX)，并且认为通过与对象的交互来利用类型空间可以促进学习可迁移的交互概念和泛化能力。这是将知识融入到强化学习中的一种自然方式，并有望成为人工智能系统中实现自主和通用行为的推广者。

    Humans and other animals aptly exhibit general intelligence behaviors in solving a variety of tasks with flexibility and ability to adapt to novel situations by reusing and applying high level knowledge acquired over time. But artificial agents are more of a specialist, lacking such generalist behaviors. Artificial agents will require understanding and exploiting critical structured knowledge representations. We present a metacognitive generalization framework, Knowledge-Interaction-eXecution (KIX), and argue that interactions with objects leveraging type space facilitate the learning of transferable interaction concepts and generalization. It is a natural way of integrating knowledge into reinforcement learning and promising to act as an enabler for autonomous and generalist behaviors in artificial intelligence systems.
    
[^3]: “任务成功”远远不够：探究将视频-语言模型作为行为批评者以捕捉不良代理行为的使用

    "Task Success" is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors

    [https://arxiv.org/abs/2402.04210](https://arxiv.org/abs/2402.04210)

    本文探究了将视频-语言模型作为行为批评者用于捕捉不良代理行为的可行性，以解决大规模生成模型忽视任务约束和用户偏好的问题。

    

    大规模生成模型被证明对于抽样有意义的候选解决方案很有用，然而它们经常忽视任务约束和用户偏好。当模型与外部验证者结合，并根据验证反馈逐步或逐渐得出最终解决方案时，它们的完全能力更好地被利用。在具身化人工智能的背景下，验证通常仅涉及评估指令中指定的目标条件是否已满足。然而，为了将这些代理者无缝地融入日常生活，必须考虑到更广泛的约束和偏好，超越仅任务成功（例如，机器人应该谨慎地抓住面包，以避免明显的变形）。然而，鉴于机器人任务的无限范围，构建类似于用于显式知识任务（如围棋和定理证明）的脚本化验证器是不可行的。这引出了一个问题：当没有可靠的验证者可用时，何时可以信任代理的行为？

    Large-scale generative models are shown to be useful for sampling meaningful candidate solutions, yet they often overlook task constraints and user preferences. Their full power is better harnessed when the models are coupled with external verifiers and the final solutions are derived iteratively or progressively according to the verification feedback. In the context of embodied AI, verification often solely involves assessing whether goal conditions specified in the instructions have been met. Nonetheless, for these agents to be seamlessly integrated into daily life, it is crucial to account for a broader range of constraints and preferences beyond bare task success (e.g., a robot should grasp bread with care to avoid significant deformations). However, given the unbounded scope of robot tasks, it is infeasible to construct scripted verifiers akin to those used for explicit-knowledge tasks like the game of Go and theorem proving. This begs the question: when no sound verifier is avail
    
[^4]: 利用强化学习和模仿学习的外科医生参与眼科机器人学徒系统研究

    Toward a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2311.17693](https://arxiv.org/abs/2311.17693)

    利用模拟图像引导的以外科医生为中心的自主机器人学徒系统，通过强化学习和模仿学习代理在眼科白内障手术中适应外科医生的技能水平和外科手术偏好。

    

    机器人辅助手术系统在提高手术精确度和减少人为错误方面展示了显著潜力。然而，现有系统缺乏适应个别外科医生的独特偏好和要求的能力。此外，它们主要集中在普通手术（如腹腔镜手术），不适用于非常精密的微创手术，如眼科手术。因此，我们提出了一种基于模拟图像引导的以外科医生为中心的自主机器人学徒系统，可在眼科白内障手术过程中适应个别外科医生的技能水平和首选外科手术技术。我们的方法利用模拟环境来训练以图像数据为指导的强化学习和模仿学习代理，以执行白内障手术的切口阶段所有任务。通过将外科医生的动作和偏好整合到训练过程中，让外科医生参与其中，我们的方法可以达到更好的效果。

    arXiv:2311.17693v2 Announce Type: replace-cross  Abstract: Robotic-assisted surgical systems have demonstrated significant potential in enhancing surgical precision and minimizing human errors. However, existing systems lack the ability to accommodate the unique preferences and requirements of individual surgeons. Additionally, they primarily focus on general surgeries (e.g., laparoscopy) and are not suitable for highly precise microsurgeries, such as ophthalmic procedures. Thus, we propose a simulation-based image-guided approach for surgeon-centered autonomous agents that can adapt to the individual surgeon's skill level and preferred surgical techniques during ophthalmic cataract surgery. Our approach utilizes a simulated environment to train reinforcement and imitation learning agents guided by image data to perform all tasks of the incision phase of cataract surgery. By integrating the surgeon's actions and preferences into the training process with the surgeon-in-the-loop, our ap
    
[^5]: 任意时间提炼的恒定时间运动规划和操纵

    Constant-time Motion Planning with Anytime Refinement for Manipulation. (arXiv:2311.00837v1 [cs.RO])

    [http://arxiv.org/abs/2311.00837](http://arxiv.org/abs/2311.00837)

    我们提出了一种任意时间提炼方法，结合恒定时间动作规划器（CTMP）在机器人操作中改善解决方案。这种方法利用机器人系统拥有的多余计划时间来完善运动规划的结果。

    

    机器人操作器对于未来的自主系统至关重要，但对其自主性的信任有限，将其限制为刚性、任务特定的系统。操作器的复杂配置空间，以及避障和约束满足的挑战经常使得运动规划成为实现可靠和适应性自主性的瓶颈。最近，引入了一类恒定时间运动规划器（CTMP）。这些规划器利用预处理阶段计算数据结构，能够在用户定义的时间限制内生成运动规划，虽然可能并不是最优解。这个框架在许多时间关键的任务中被证明是有效的。然而，机器人系统通常有比CTMP所需的在线部分更多的计划时间，这些时间可以用来改善解决方案。为此，我们提出了一个在CTMP中与任意时间提炼方法结合使用的方法。

    Robotic manipulators are essential for future autonomous systems, yet limited trust in their autonomy has confined them to rigid, task-specific systems. The intricate configuration space of manipulators, coupled with the challenges of obstacle avoidance and constraint satisfaction, often makes motion planning the bottleneck for achieving reliable and adaptable autonomy. Recently, a class of constant-time motion planners (CTMP) was introduced. These planners employ a preprocessing phase to compute data structures that enable online planning provably guarantee the ability to generate motion plans, potentially sub-optimal, within a user defined time bound. This framework has been demonstrated to be effective in a number of time-critical tasks. However, robotic systems often have more time allotted for planning than the online portion of CTMP requires, time that can be used to improve the solution. To this end, we propose an anytime refinement approach that works in combination with CTMP a
    
[^6]: CoFiI2P: 粗到精的图像到点云注册的对应关系

    CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration. (arXiv:2309.14660v1 [cs.CV])

    [http://arxiv.org/abs/2309.14660](http://arxiv.org/abs/2309.14660)

    CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。

    

    图像到点云（I2P）注册是机器人导航和移动建图领域中的一项基础任务。现有的I2P注册方法在点到像素级别上估计对应关系，忽略了全局对齐。然而，没有来自全局约束的高级引导的I2P匹配容易收敛到局部最优解。为了解决这个问题，本文提出了一种新的I2P注册网络CoFiI2P，通过粗到精的方式提取对应关系，以得到全局最优解。首先，将图像和点云输入到一个共享编码-解码网络中进行层次化特征提取。然后，设计了一个粗到精的匹配模块，利用特征建立稳健的特征对应关系。具体来说，在粗匹配块中，采用了一种新型的I2P变换模块，从图像和点云中捕捉同质和异质的全局信息。通过判别描述子，完成粗-细特征匹配过程。最后，通过细化匹配模块进一步提升对应关系的准确性。

    Image-to-point cloud (I2P) registration is a fundamental task in the fields of robot navigation and mobile mapping. Existing I2P registration works estimate correspondences at the point-to-pixel level, neglecting the global alignment. However, I2P matching without high-level guidance from global constraints may converge to the local optimum easily. To solve the problem, this paper proposes CoFiI2P, a novel I2P registration network that extracts correspondences in a coarse-to-fine manner for the global optimal solution. First, the image and point cloud are fed into a Siamese encoder-decoder network for hierarchical feature extraction. Then, a coarse-to-fine matching module is designed to exploit features and establish resilient feature correspondences. Specifically, in the coarse matching block, a novel I2P transformer module is employed to capture the homogeneous and heterogeneous global information from image and point cloud. With the discriminate descriptors, coarse super-point-to-su
    

