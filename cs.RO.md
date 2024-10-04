# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation](https://arxiv.org/abs/2403.19460) | RiEMann提出了一个近实时SE(3)等变机器人操作模仿学习框架，无需点云分割，可以从零开始学习操作任务，泛化到看不见的转换和目标对象实例，对抗视觉干扰，实时跟踪目标对象的姿势变化，同时具有可扩展的动作空间使得关节对象操作成为可能。 |
| [^2] | [CMP: Cooperative Motion Prediction with Multi-Agent Communication](https://arxiv.org/abs/2403.17916) | 该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。 |
| [^3] | [Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks](https://arxiv.org/abs/2403.14488) | 该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。 |
| [^4] | [Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning](https://arxiv.org/abs/2402.14407) | 利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。 |
| [^5] | [PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Preference Alignment](https://arxiv.org/abs/2402.08702) | 该论文介绍了一种在多步任务中集成人类反馈和偏好对齐的PRompt优化方法。它使用遗传算法框架，结合人类反馈自动提出优化建议并解决了复杂的提示内容分析、单步评估和任务执行偏好的挑战。 |
| [^6] | [Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance.](http://arxiv.org/abs/2310.02635) | 本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。 |

# 详细

[^1]: RiEMann: 不需要点云分割的近实时 SE(3)等变机器人操作

    RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation

    [https://arxiv.org/abs/2403.19460](https://arxiv.org/abs/2403.19460)

    RiEMann提出了一个近实时SE(3)等变机器人操作模仿学习框架，无需点云分割，可以从零开始学习操作任务，泛化到看不见的转换和目标对象实例，对抗视觉干扰，实时跟踪目标对象的姿势变化，同时具有可扩展的动作空间使得关节对象操作成为可能。

    

    我们提出了RiEMann，一个端到端的近实时 SE(3)等变机器人操作模仿学习框架，从场景点云输入中学习。与先前依赖描述符匹配的方法不同，RiEMann直接预测对象的目标姿势进行操作，而无需进行任何对象分割。RiEMann可以从零开始学习一个操作任务，只需5到10个演示，可以泛化到看不见的SE(3)转换和目标对象的实例，抵抗干扰对象的视觉干扰，并遵循目标对象的近实时姿势变化。RiEMann的可伸缩动作空间有助于添加自定义等变动作，例如旋转水龙头的方向，这使得RiEMann可以进行关节对象操作。在模拟和现实世界的6自由度机器人操作实验中，我们测试了RiEMann 对 5类操纵任务的25种变体进行了测试。

    arXiv:2403.19460v1 Announce Type: cross  Abstract: We present RiEMann, an end-to-end near Real-time SE(3)-Equivariant Robot Manipulation imitation learning framework from scene point cloud input. Compared to previous methods that rely on descriptor field matching, RiEMann directly predicts the target poses of objects for manipulation without any object segmentation. RiEMann learns a manipulation task from scratch with 5 to 10 demonstrations, generalizes to unseen SE(3) transformations and instances of target objects, resists visual interference of distracting objects, and follows the near real-time pose change of the target object. The scalable action space of RiEMann facilitates the addition of custom equivariant actions such as the direction of turning the faucet, which makes articulated object manipulation possible for RiEMann. In simulation and real-world 6-DOF robot manipulation experiments, we test RiEMann on 5 categories of manipulation tasks with a total of 25 variants and show
    
[^2]: CMP：具有多智能体通信的合作运动预测

    CMP: Cooperative Motion Prediction with Multi-Agent Communication

    [https://arxiv.org/abs/2403.17916](https://arxiv.org/abs/2403.17916)

    该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。

    

    随着自动驾驶车辆（AVs）的发展和车联网（V2X）通信的成熟，合作连接的自动化车辆（CAVs）的功能变得可能。本文基于合作感知，探讨了合作运动预测的可行性和有效性。我们的方法CMP以LiDAR信号作为输入，以增强跟踪和预测能力。与过去专注于合作感知或运动预测的工作不同，我们的框架是我们所知的第一个解决CAVs在感知和预测模块中共享信息的统一问题。我们的设计中还融入了能够容忍现实V2X带宽限制和传输延迟的独特能力，同时处理庞大的感知表示。我们还提出了预测聚合模块，统一了预测

    arXiv:2403.17916v1 Announce Type: cross  Abstract: The confluence of the advancement of Autonomous Vehicles (AVs) and the maturity of Vehicle-to-Everything (V2X) communication has enabled the capability of cooperative connected and automated vehicles (CAVs). Building on top of cooperative perception, this paper explores the feasibility and effectiveness of cooperative motion prediction. Our method, CMP, takes LiDAR signals as input to enhance tracking and prediction capabilities. Unlike previous work that focuses separately on either cooperative perception or motion prediction, our framework, to the best of our knowledge, is the first to address the unified problem where CAVs share information in both perception and prediction modules. Incorporated into our design is the unique capability to tolerate realistic V2X bandwidth limitations and transmission delays, while dealing with bulky perception representations. We also propose a prediction aggregation module, which unifies the predict
    
[^3]: 基于物理学因果推理的机器人操作任务中安全稳健的下一最佳动作选择

    Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks

    [https://arxiv.org/abs/2403.14488](https://arxiv.org/abs/2403.14488)

    该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。

    

    安全高效的物体操作是许多真实世界机器人应用的关键推手。然而，这种挑战在于机器人操作必须对一系列传感器和执行器的不确定性具有稳健性。本文提出了一个基于物理知识和因果推理的框架，用于让机器人在部分可观察的环境中对候选动作进行概率推理，以完成一个积木堆叠任务。我们将刚体系统动力学的基于物理学的仿真与因果贝叶斯网络（CBN）结合起来，定义了机器人决策过程的因果生成概率模型。通过基于仿真的蒙特卡洛实验，我们展示了我们的框架成功地能够：(1) 高准确度地预测积木塔的稳定性（预测准确率：88.6%）；和，(2) 为积木堆叠任务选择一个近似的下一最佳动作，供整合的机器人系统执行，实现94.2%的任务成功率。

    arXiv:2403.14488v1 Announce Type: cross  Abstract: Safe and efficient object manipulation is a key enabler of many real-world robot applications. However, this is challenging because robot operation must be robust to a range of sensor and actuator uncertainties. In this paper, we present a physics-informed causal-inference-based framework for a robot to probabilistically reason about candidate actions in a block stacking task in a partially observable setting. We integrate a physics-based simulation of the rigid-body system dynamics with a causal Bayesian network (CBN) formulation to define a causal generative probabilistic model of the robot decision-making process. Using simulation-based Monte Carlo experiments, we demonstrate our framework's ability to successfully: (1) predict block tower stability with high accuracy (Pred Acc: 88.6%); and, (2) select an approximate next-best action for the block stacking task, for execution by an integrated robot system, achieving 94.2% task succe
    
[^4]: 通过离散扩散进行大规模无动作视频预训练，以实现高效策略学习

    Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning

    [https://arxiv.org/abs/2402.14407](https://arxiv.org/abs/2402.14407)

    利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。

    

    学习一个能够完成多个任务的通用实体代理面临挑战，主要源自缺乏有标记动作的机器人数据集。相比之下，存在大量捕捉复杂任务和与物理世界互动的人类视频。本文介绍了一种新颖框架，利用统一的离散扩散将人类视频上的生成式预训练与少量有标记机器人视频上的策略微调结合起来。我们首先将人类和机器人视频压缩成统一的视频标记。在预训练阶段，我们使用一个带有蒙版替换扩散策略的离散扩散模型来预测潜空间中的未来视频标记。在微调阶段，我们 h

    arXiv:2402.14407v1 Announce Type: new  Abstract: Learning a generalist embodied agent capable of completing multiple tasks poses challenges, primarily stemming from the scarcity of action-labeled robotic datasets. In contrast, a vast amount of human videos exist, capturing intricate tasks and interactions with the physical world. Promising prospects arise for utilizing actionless human videos for pre-training and transferring the knowledge to facilitate robot policy learning through limited robot demonstrations. In this paper, we introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We start by compressing both human and robot videos into unified video tokens. In the pre-training stage, we employ a discrete diffusion model with a mask-and-replace diffusion strategy to predict future video tokens in the latent space. In the fine-tuning stage, we h
    
[^5]: PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Preference Alignment

    PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Preference Alignment

    [https://arxiv.org/abs/2402.08702](https://arxiv.org/abs/2402.08702)

    该论文介绍了一种在多步任务中集成人类反馈和偏好对齐的PRompt优化方法。它使用遗传算法框架，结合人类反馈自动提出优化建议并解决了复杂的提示内容分析、单步评估和任务执行偏好的挑战。

    

    Prompt optimization aims to find the best prompt for a language model (LLM) in multi-step tasks. This paper introduces a genetic algorithm framework that incorporates human feedback to automatically suggest prompt improvements. It addresses challenges such as complex prompt content analysis, evaluation of individual steps, and variations in task execution preferences.

    arXiv:2402.08702v1 Announce Type: cross Abstract: Prompt optimization aims to find the best prompt to a large language model (LLM) for a given task. LLMs have been successfully used to help find and improve prompt candidates for single-step tasks. However, realistic tasks for agents are multi-step and introduce new challenges: (1) Prompt content is likely to be more extensive and complex, making it more difficult for LLMs to analyze errors, (2) the impact of an individual step is difficult to evaluate, and (3) different people may have varied preferences about task execution. While humans struggle to optimize prompts, they are good at providing feedback about LLM outputs; we therefore introduce a new LLM-driven discrete prompt optimization framework that incorporates human-designed feedback rules about potential errors to automatically offer direct suggestions for improvement. Our framework is stylized as a genetic algorithm in which an LLM generates new candidate prompts from a parent
    
[^6]: 强化学习基础：朝向具有基础先验辅助的具身通用智能体

    Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance. (arXiv:2310.02635v1 [cs.RO])

    [http://arxiv.org/abs/2310.02635](http://arxiv.org/abs/2310.02635)

    本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。

    

    最近人们已经表明，从互联网规模的数据中进行大规模预训练是构建通用模型的关键，正如在NLP中所见。为了构建具身通用智能体，我们和许多其他研究者假设这种基础先验也是不可或缺的组成部分。然而，目前尚不清楚如何以适当的具体形式表示这些具身基础先验，以及它们应该如何在下游任务中使用。在本文中，我们提出了一组直观有效的具身先验，包括基础策略、价值和成功奖励。所提出的先验是基于目标条件的MDP。为了验证其有效性，我们实例化了一个由这些先验辅助的演员-评论家方法，称之为基础演员-评论家（FAC）。我们将我们的框架命名为基础强化学习（FRL），因为它完全依赖于具身基础先验来进行探索、学习和强化。FRL的好处有三个。(1)样本效率高。通过基础先验加速训练过程，减少样本使用量。

    Recently, people have shown that large-scale pre-training from internet-scale data is the key to building generalist models, as witnessed in NLP. To build embodied generalist agents, we and many other researchers hypothesize that such foundation prior is also an indispensable component. However, it is unclear what is the proper concrete form to represent those embodied foundation priors and how they should be used in the downstream task. In this paper, we propose an intuitive and effective set of embodied priors that consist of foundation policy, value, and success reward. The proposed priors are based on the goal-conditioned MDP. To verify their effectiveness, we instantiate an actor-critic method assisted by the priors, called Foundation Actor-Critic (FAC). We name our framework as Foundation Reinforcement Learning (FRL), since it completely relies on embodied foundation priors to explore, learn and reinforce. The benefits of FRL are threefold. (1) Sample efficient. With foundation p
    

