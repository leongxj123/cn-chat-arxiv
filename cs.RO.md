# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation](https://arxiv.org/abs/2402.15487) | 本文提出了交互式场景探索任务，通过自主探索环境生成了动作条件化场景图，捕捉了环境的结构 |
| [^2] | [Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning](https://arxiv.org/abs/2402.14407) | 利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。 |
| [^3] | [Feudal Networks for Visual Navigation](https://arxiv.org/abs/2402.12498) | 使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。 |
| [^4] | [Two is Better Than One: Digital Siblings to Improve Autonomous Driving Testing.](http://arxiv.org/abs/2305.08060) | 本文提出了数字孪生的概念，使用不同技术构建多个通用仿真器，强化了自动驾驶软件的基于仿真的测试，提高了测试结果的普适性和可靠性。 |

# 详细

[^1]: RoboEXP: 通过交互式探索实现动作条件化场景图用于机器人操作

    RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation

    [https://arxiv.org/abs/2402.15487](https://arxiv.org/abs/2402.15487)

    本文提出了交互式场景探索任务，通过自主探索环境生成了动作条件化场景图，捕捉了环境的结构

    

    机器人需要探索周围环境以适应并应对未知环境中的任务。本文介绍了交互式场景探索的新任务，其中机器人自主探索环境并生成一个捕捉基础环境结构的动作条件化场景图（ACSG）

    arXiv:2402.15487v1 Announce Type: cross  Abstract: Robots need to explore their surroundings to adapt to and tackle tasks in unknown environments. Prior work has proposed building scene graphs of the environment but typically assumes that the environment is static, omitting regions that require active interactions. This severely limits their ability to handle more complex tasks in household and office environments: before setting up a table, robots must explore drawers and cabinets to locate all utensils and condiments. In this work, we introduce the novel task of interactive scene exploration, wherein robots autonomously explore environments and produce an action-conditioned scene graph (ACSG) that captures the structure of the underlying environment. The ACSG accounts for both low-level information, such as geometry and semantics, and high-level information, such as the action-conditioned relationships between different entities in the scene. To this end, we present the Robotic Explo
    
[^2]: 通过离散扩散进行大规模无动作视频预训练，以实现高效策略学习

    Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning

    [https://arxiv.org/abs/2402.14407](https://arxiv.org/abs/2402.14407)

    利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。

    

    学习一个能够完成多个任务的通用实体代理面临挑战，主要源自缺乏有标记动作的机器人数据集。相比之下，存在大量捕捉复杂任务和与物理世界互动的人类视频。本文介绍了一种新颖框架，利用统一的离散扩散将人类视频上的生成式预训练与少量有标记机器人视频上的策略微调结合起来。我们首先将人类和机器人视频压缩成统一的视频标记。在预训练阶段，我们使用一个带有蒙版替换扩散策略的离散扩散模型来预测潜空间中的未来视频标记。在微调阶段，我们 h

    arXiv:2402.14407v1 Announce Type: new  Abstract: Learning a generalist embodied agent capable of completing multiple tasks poses challenges, primarily stemming from the scarcity of action-labeled robotic datasets. In contrast, a vast amount of human videos exist, capturing intricate tasks and interactions with the physical world. Promising prospects arise for utilizing actionless human videos for pre-training and transferring the knowledge to facilitate robot policy learning through limited robot demonstrations. In this paper, we introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We start by compressing both human and robot videos into unified video tokens. In the pre-training stage, we employ a discrete diffusion model with a mask-and-replace diffusion strategy to predict future video tokens in the latent space. In the fine-tuning stage, we h
    
[^3]: 封建网络用于视觉导航

    Feudal Networks for Visual Navigation

    [https://arxiv.org/abs/2402.12498](https://arxiv.org/abs/2402.12498)

    使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。

    

    视觉导航遵循人类可以在没有详细地图的情况下导航的直觉。一种常见方法是在建立包含可用于规划的图像节点的拓扑图的同时进行交互式探索。最近的变体从被动视频中学习，并可以利用复杂的社交和语义线索进行导航。然而，需要大量的训练视频，利用大型图并且由于使用了里程计，场景不是未知的。我们引入了一种使用封建学习的视觉导航的新方法，该方法采用了由工作代理、中级管理者和高级管理者组成的分层结构。封建学习范式的关键在于，每个级别的代理看到任务的不同方面，并且在不同的空间和时间尺度上运作。在此框架中开发了两个独特的模块。对于高级管理者，我们自监督地学习一个记忆代理地图以记录

    arXiv:2402.12498v1 Announce Type: cross  Abstract: Visual navigation follows the intuition that humans can navigate without detailed maps. A common approach is interactive exploration while building a topological graph with images at nodes that can be used for planning. Recent variations learn from passive videos and can navigate using complex social and semantic cues. However, a significant number of training videos are needed, large graphs are utilized, and scenes are not unseen since odometry is utilized. We introduce a new approach to visual navigation using feudal learning, which employs a hierarchical structure consisting of a worker agent, a mid-level manager, and a high-level manager. Key to the feudal learning paradigm, agents at each level see a different aspect of the task and operate at different spatial and temporal scales. Two unique modules are developed in this framework. For the high- level manager, we learn a memory proxy map in a self supervised manner to record prio
    
[^4]: 两个优于一个：数字孪生以提高自动驾驶测试

    Two is Better Than One: Digital Siblings to Improve Autonomous Driving Testing. (arXiv:2305.08060v1 [cs.SE])

    [http://arxiv.org/abs/2305.08060](http://arxiv.org/abs/2305.08060)

    本文提出了数字孪生的概念，使用不同技术构建多个通用仿真器，强化了自动驾驶软件的基于仿真的测试，提高了测试结果的普适性和可靠性。

    

    基于仿真的测试是确保自动驾驶软件可靠性的重要一步。实际中，当企业依赖第三方通用仿真器进行内部或外包测试时，测试结果的普适性受到威胁。在本文中，我们通过引入“数字孪生”的概念加强了基于仿真的测试，这是一个新颖的框架，在其中AV在多个使用不同技术构建的通用仿真器上进行测试。首先，针对每个单独的仿真器自动生成测试用例。然后，使用特征映射将测试迁移至各个仿真器之间，以表征所进行的行驶条件。最后，计算联合预测失效概率，并仅在孪生之间达成一致的情况下报告故障。我们使用两个开源仿真器实现了该框架，并在数字孪生的物理比例模型上进行了经验比较。

    Simulation-based testing represents an important step to ensure the reliability of autonomous driving software. In practice, when companies rely on third-party general-purpose simulators, either for in-house or outsourced testing, the generalizability of testing results to real autonomous vehicles is at stake.  In this paper, we strengthen simulation-based testing by introducing the notion of digital siblings, a novel framework in which the AV is tested on multiple general-purpose simulators, built with different technologies. First, test cases are automatically generated for each individual simulator. Then, tests are migrated between simulators, using feature maps to characterize of the exercised driving conditions. Finally, the joint predicted failure probability is computed and a failure is reported only in cases of agreement among the siblings.  We implemented our framework using two open-source simulators and we empirically compared it against a digital twin of a physical scaled a
    

