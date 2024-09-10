# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics](https://arxiv.org/abs/2403.19578) | 使用关键动作令牌（KAT）框架，研究展示了文本预训练的变形器（GPT-4 Turbo）在机器人领域可实现视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列，表现优越于现有的模仿学习方法。 |
| [^2] | [LeTac-MPC: Learning Model Predictive Control for Tactile-reactive Grasping](https://arxiv.org/abs/2403.04934) | LeTac-MPC是一种学习模型预测控制，利用视觉触觉传感器GelSight和不同iable MPC层，实现在不同条件下和具有不同物理属性的物体上进行稳健抓取控制。 |
| [^3] | [Learning Generalizable Tool-use Skills through Trajectory Generation.](http://arxiv.org/abs/2310.00156) | 通过轨迹生成，我们提出了一种学习通用工具使用技能的方法，可以适应不同形状的工具，从而使自主系统能够处理复杂的可变形物体操作任务。 |

# 详细

[^1]: 关键动作令牌在机器人学中实现上下文模仿学习

    Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics

    [https://arxiv.org/abs/2403.19578](https://arxiv.org/abs/2403.19578)

    使用关键动作令牌（KAT）框架，研究展示了文本预训练的变形器（GPT-4 Turbo）在机器人领域可实现视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列，表现优越于现有的模仿学习方法。

    

    我们展示了现成的基于文本的变形器，无需额外训练，就可以执行少样本上下文内视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列。我们通过将视觉观测（输入）和动作轨迹（输出）转换为一系列令牌，这些令牌可以被文本预训练的变形器（GPT-4 Turbo）接收和生成，通过我们称之为关键动作令牌（KAT）的框架来实现这一点。尽管仅在语言上训练，我们展示这些变形器擅长将标记化的视觉关键点观察翻译为行为轨迹，在真实世界的日常任务套件中，在低数据情况下表现与优于最先进的模仿学习（扩散策略）。KAT不同于通常在语言领域操作，它利用基于文本的变形器在视觉和动作领域中学习。

    arXiv:2403.19578v1 Announce Type: cross  Abstract: We show that off-the-shelf text-based Transformers, with no additional training, can perform few-shot in-context visual imitation learning, mapping visual observations to action sequences that emulate the demonstrator's behaviour. We achieve this by transforming visual observations (inputs) and trajectories of actions (outputs) into sequences of tokens that a text-pretrained Transformer (GPT-4 Turbo) can ingest and generate, via a framework we call Keypoint Action Tokens (KAT). Despite being trained only on language, we show that these Transformers excel at translating tokenised visual keypoint observations into action trajectories, performing on par or better than state-of-the-art imitation learning (diffusion policies) in the low-data regime on a suite of real-world, everyday tasks. Rather than operating in the language domain as is typical, KAT leverages text-based Transformers to operate in the vision and action domains to learn ge
    
[^2]: LeTac-MPC：用于触觉反应抓取的学习模型预测控制

    LeTac-MPC: Learning Model Predictive Control for Tactile-reactive Grasping

    [https://arxiv.org/abs/2403.04934](https://arxiv.org/abs/2403.04934)

    LeTac-MPC是一种学习模型预测控制，利用视觉触觉传感器GelSight和不同iable MPC层，实现在不同条件下和具有不同物理属性的物体上进行稳健抓取控制。

    

    抓取是机器人中的关键任务，需要触觉反馈和反应性抓取调整，以实现在各种条件下和具有不同物理属性的对象的稳健抓取。本文介绍了LeTac-MPC，一种基于学习的模型预测控制（MPC）用于触觉反应式抓取。我们的方法使夹爪能够在动态和力交互任务中抓取具有不同物理属性的对象。我们利用基于视觉的触觉传感器GelSight，该传感器能够感知包含抓取对象的物理属性和状态信息的高分辨率触觉反馈。LeTac-MPC包含一个可微分的MPC层，设计用于对通过神经网络（NN）从触觉反馈中提取的嵌入进行建模。这种设计有助于在25 Hz的频率下实现收敛和稳健的抓取控制。我们提出了一个完全自动化的数据收集流程，并收集了一组数据集。

    arXiv:2403.04934v1 Announce Type: cross  Abstract: Grasping is a crucial task in robotics, necessitating tactile feedback and reactive grasping adjustments for robust grasping of objects under various conditions and with differing physical properties. In this paper, we introduce LeTac-MPC, a learning-based model predictive control (MPC) for tactile-reactive grasping. Our approach enables the gripper grasp objects with different physical properties on dynamic and force-interactive tasks. We utilize a vision-based tactile sensor, GelSight, which is capable of perceiving high-resolution tactile feedback that contains the information of physical properties and states of the grasped object. LeTac-MPC incorporates a differentiable MPC layer designed to model the embeddings extracted by a neural network (NN) from tactile feedback. This design facilitates convergent and robust grasping control at a frequency of 25 Hz. We propose a fully automated data collection pipeline and collect a dataset 
    
[^3]: 通过轨迹生成学习具有通用性的工具使用技能

    Learning Generalizable Tool-use Skills through Trajectory Generation. (arXiv:2310.00156v1 [cs.RO])

    [http://arxiv.org/abs/2310.00156](http://arxiv.org/abs/2310.00156)

    通过轨迹生成，我们提出了一种学习通用工具使用技能的方法，可以适应不同形状的工具，从而使自主系统能够处理复杂的可变形物体操作任务。

    

    高效利用工具的自主系统可以帮助人们完成许多常见任务，如烹饪和清洁。然而，当前的系统在适应新工具方面远远不及人类的智能水平。基于可及性的先前工作通常对环境做出了很强的假设，并且无法扩展到更复杂、接触丰富的任务。 在这项工作中，我们解决了这个挑战，并探索了代理如何学习使用以前未见过的工具来操纵可变形物体。 我们提出了将工具使用轨迹作为一系列点云的生成模型，可以推广到不同的工具形状。对于任何新的工具，我们首先生成一个工具使用轨迹，然后优化工具姿势序列以与生成的轨迹对齐。我们为四种不同的具有挑战性的可变形物体操纵任务训练了一个单一模型。我们的模型仅使用每个任务的单个工具的示范数据进行训练，并且能够...

    Autonomous systems that efficiently utilize tools can assist humans in completing many common tasks such as cooking and cleaning. However, current systems fall short of matching human-level of intelligence in terms of adapting to novel tools. Prior works based on affordance often make strong assumptions about the environments and cannot scale to more complex, contact-rich tasks. In this work, we tackle this challenge and explore how agents can learn to use previously unseen tools to manipulate deformable objects. We propose to learn a generative model of the tool-use trajectories as a sequence of point clouds, which generalizes to different tool shapes. Given any novel tool, we first generate a tool-use trajectory and then optimize the sequence of tool poses to align with the generated trajectory. We train a single model for four different challenging deformable object manipulation tasks. Our model is trained with demonstration data from just a single tool for each task and is able to 
    

