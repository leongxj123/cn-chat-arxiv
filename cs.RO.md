# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Guessing human intentions to avoid dangerous situations in caregiving robots](https://arxiv.org/abs/2403.16291) | 本文探讨了在照料机器人中使用人工心智理论来猜测人类意图，提出了一种检测危险情况并实时消除危险的算法，在模拟实验中取得了高成功率。 |
| [^2] | [RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems](https://arxiv.org/abs/2403.12853) | 提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。 |
| [^3] | [Ensembling Prioritized Hybrid Policies for Multi-agent Pathfinding](https://arxiv.org/abs/2403.07559) | 提出了Ensembling Prioritized Hybrid Policies (EPH)方法，通过选择性通信模块和三种高级推理策略，提高了基于通信的多智能体路径规划解决方案的性能。 |
| [^4] | [Learning Agility Adaptation for Flight in Clutter](https://arxiv.org/abs/2403.04586) | 本文旨在使飞行器在未知且部分可观测的杂乱环境中具有敏捷性调整能力，提出了一种利用分层学习和规划框架的方法，通过在线无模型强化学习和预训练微调奖励方案获得可部署的策略，在仿真中显示出比恒定敏捷性基线和另一种替代方法更优越的飞行效率和安全性。 |
| [^5] | [Towards scalable robotic intervention of children with Autism Spectrum Disorder using LLMs](https://arxiv.org/abs/2402.00260) | 本文提出了一种以Large Language Model (LLM)为基础的社交机器人，用于与自闭症谱系障碍儿童进行口头交流，教授透视能力。通过比较不同的LLM管道，发现GPT-2 + BART管道可以更好地生成问题和选择项。这种研究有助于改善自闭症儿童的社交能力。 |
| [^6] | [Left/Right Brain, human motor control and the implications for robotics.](http://arxiv.org/abs/2401.14057) | 本研究通过训练不同的损失函数，实现了类似于人类的左右半球专门化控制系统，该系统在不同的运动任务中展现出协调性、运动效率和位置稳定性的优势。 |

# 详细

[^1]: 在照料机器人中猜测人类意图以避免危险情况

    Guessing human intentions to avoid dangerous situations in caregiving robots

    [https://arxiv.org/abs/2403.16291](https://arxiv.org/abs/2403.16291)

    本文探讨了在照料机器人中使用人工心智理论来猜测人类意图，提出了一种检测危险情况并实时消除危险的算法，在模拟实验中取得了高成功率。

    

    要求机器人进行社交互动，它们必须准确解释人类意图并预测潜在结果。对于为人类护理设计的社交机器人而言尤为重要，可能会面临人类的危险情况，比如未见障碍物，应该予以避免。本文探讨了人工心智理论（ATM）方法来推断和解释人类意图。我们提出了一种检测人类风险情况的算法，选择实时消除危险的机器人动作。我们采用基于模拟的ATM方法，并采用“像我一样”的策略将意图和动作分配给人类。通过这种策略，机器人在有限时间内可以高成功率地检测和行动。该算法已经作为现有机器人认知架构的一部分实施，并在模拟场景中进行了测试。进行了三个实验。

    arXiv:2403.16291v1 Announce Type: cross  Abstract: For robots to interact socially, they must interpret human intentions and anticipate their potential outcomes accurately. This is particularly important for social robots designed for human care, which may face potentially dangerous situations for people, such as unseen obstacles in their way, that should be avoided. This paper explores the Artificial Theory of Mind (ATM) approach to inferring and interpreting human intentions. We propose an algorithm that detects risky situations for humans, selecting a robot action that removes the danger in real time. We use the simulation-based approach to ATM and adopt the 'like-me' policy to assign intentions and actions to people. Using this strategy, the robot can detect and act with a high rate of success under time-constrained situations. The algorithm has been implemented as part of an existing robotics cognitive architecture and tested in simulation scenarios. Three experiments have been co
    
[^2]: 基于无人机的环境智能系统的可重构作动和传感平台RASP

    RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems

    [https://arxiv.org/abs/2403.12853](https://arxiv.org/abs/2403.12853)

    提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。

    

    实现消费级无人机与我们家中的吸尘机器人或日常生活中的个人智能手机一样有用，需要无人机能感知、驱动和响应可能出现的一般情况。为了实现这一愿景，我们提出了RASP，一个模块化和可重构的传感和作动平台，允许无人机在仅25秒内自主更换机载传感器和执行器，使单个无人机能够快速适应各种任务。RASP包括一个机械层，用于物理更换传感器模块，一个电气层，用于维护传感器/执行器的电源和通信线路，以及一个软件层，用于在无人机和我们平台上的任何传感器模块之间维护一个公共接口。利用最近在大型语言和视觉语言模型方面的进展，我们进一步介绍了一种利用RASP的个人助理系统的架构、实现和现实世界部署。

    arXiv:2403.12853v1 Announce Type: cross  Abstract: Realizing consumer-grade drones that are as useful as robot vacuums throughout our homes or personal smartphones in our daily lives requires drones to sense, actuate, and respond to general scenarios that may arise. Towards this vision, we propose RASP, a modular and reconfigurable sensing and actuation platform that allows drones to autonomously swap onboard sensors and actuators in only 25 seconds, allowing a single drone to quickly adapt to a diverse range of tasks. RASP consists of a mechanical layer to physically swap sensor modules, an electrical layer to maintain power and communication lines to the sensor/actuator, and a software layer to maintain a common interface between the drone and any sensor module in our platform. Leveraging recent advances in large language and visual language models, we further introduce the architecture, implementation, and real-world deployments of a personal assistant system utilizing RASP. We demo
    
[^3]: 为多智能体路径规划集成优先级混合策略

    Ensembling Prioritized Hybrid Policies for Multi-agent Pathfinding

    [https://arxiv.org/abs/2403.07559](https://arxiv.org/abs/2403.07559)

    提出了Ensembling Prioritized Hybrid Policies (EPH)方法，通过选择性通信模块和三种高级推理策略，提高了基于通信的多智能体路径规划解决方案的性能。

    

    基于多智能体强化学习（MARL）的多智能体路径规划（MAPF）近来因其高效性和可扩展性而受到关注。我们提出了一种新方法，Ensembling Prioritized Hybrid Policies (EPH)，以进一步提高基于通信的MARL-MAPF求解器的性能。我们首先提出了一个选择性通信模块，以在多智能体环境中收集更丰富的信息，从而实现更好的智能体协调，并使用基于Q-learning的算法对模型进行训练。

    arXiv:2403.07559v1 Announce Type: cross  Abstract: Multi-Agent Reinforcement Learning (MARL) based Multi-Agent Path Finding (MAPF) has recently gained attention due to its efficiency and scalability. Several MARL-MAPF methods choose to use communication to enrich the information one agent can perceive. However, existing works still struggle in structured environments with high obstacle density and a high number of agents. To further improve the performance of the communication-based MARL-MAPF solvers, we propose a new method, Ensembling Prioritized Hybrid Policies (EPH). We first propose a selective communication block to gather richer information for better agent coordination within multi-agent environments and train the model with a Q-learning-based algorithm. We further introduce three advanced inference strategies aimed at bolstering performance during the execution phase. First, we hybridize the neural policy with single-agent expert guidance for navigating conflict-free zones. Se
    
[^4]: 在杂乱环境中学习飞行敏捷性调整

    Learning Agility Adaptation for Flight in Clutter

    [https://arxiv.org/abs/2403.04586](https://arxiv.org/abs/2403.04586)

    本文旨在使飞行器在未知且部分可观测的杂乱环境中具有敏捷性调整能力，提出了一种利用分层学习和规划框架的方法，通过在线无模型强化学习和预训练微调奖励方案获得可部署的策略，在仿真中显示出比恒定敏捷性基线和另一种替代方法更优越的飞行效率和安全性。

    

    动物学习适应其运动能力和操作环境的敏捷性。移动机器人也应展示这种能力，将敏捷性和安全性结合起来。本文旨在赋予飞行器在未知且部分可观测的杂乱环境中适应敏捷性的能力。我们提出了一种分层学习和规划框架，结合试错学习和基于模型的轨迹生成方法来全面学习敏捷性策略。我们使用在线无模型强化学习和预训练微调奖励方案来获得可部署的策略。在仿真中的统计结果显示，相较于恒定敏捷性基线和另一种替代方法，我们的方法在飞行效率和安全性方面具有优势。特别是，该策略导致

    arXiv:2403.04586v1 Announce Type: cross  Abstract: Animals learn to adapt agility of their movements to their capabilities and the environment they operate in. Mobile robots should also demonstrate this ability to combine agility and safety. The aim of this work is to endow flight vehicles with the ability of agility adaptation in prior unknown and partially observable cluttered environments. We propose a hierarchical learning and planning framework where we utilize both trial and error to comprehensively learn an agility policy with the vehicle's observation as the input, and well-established methods of model-based trajectory generation. Technically, we use online model-free reinforcement learning and a pre-training-fine-tuning reward scheme to obtain the deployable policy. The statistical results in simulation demonstrate the advantages of our method over the constant agility baselines and an alternative method in terms of flight efficiency and safety. In particular, the policy leads
    
[^5]: 以LLM为基础实现面向自闭症谱系障碍儿童的可扩展机器人干预

    Towards scalable robotic intervention of children with Autism Spectrum Disorder using LLMs

    [https://arxiv.org/abs/2402.00260](https://arxiv.org/abs/2402.00260)

    本文提出了一种以Large Language Model (LLM)为基础的社交机器人，用于与自闭症谱系障碍儿童进行口头交流，教授透视能力。通过比较不同的LLM管道，发现GPT-2 + BART管道可以更好地生成问题和选择项。这种研究有助于改善自闭症儿童的社交能力。

    

    本文提出了一种能够与自闭症谱系障碍(ASD)儿童进行口头交流的社交机器人。这种交流旨在通过使用Large Language Model (LLM)生成的文本来教授透视能力。社交机器人NAO扮演了一个刺激器(口头描述社交情景并提问)、提示器(提供三个选择项供选择)和奖励器(当答案正确时给予称赞)的角色。对于刺激器的角色，社交情境、问题和选择项是使用我们的LLM管道生成的。我们比较了两种方法：GPT-2 + BART和GPT-2 + GPT-2，其中第一个GPT-2在管道中是用于无监督社交情境生成的。我们使用SOCIALIQA数据集对所有LLM管道进行微调。我们发现，GPT-2 + BART管道在通过结合各自的损失函数来生成问题和选择项方面具有较好的BERTscore。这种观察结果也与儿童在交互过程中的合作水平一致。

    In this paper, we propose a social robot capable of verbally interacting with children with Autism Spectrum Disorder (ASD). This communication is meant to teach perspective-taking using text generated using a Large Language Model (LLM) pipeline. The social robot NAO acts as a stimulator (verbally describes a social situation and asks a question), prompter (presents three options to choose from), and reinforcer (praises when the answer is correct). For the role of the stimulator, the social situation, questions, and options are generated using our LLM pipeline. We compare two approaches: GPT-2 + BART and GPT-2 + GPT-2, where the first GPT-2 common between the pipelines is used for unsupervised social situation generation. We use the SOCIALIQA dataset to fine-tune all of our LLM pipelines. We found that the GPT-2 + BART pipeline had a better BERTscore for generating the questions and the options by combining their individual loss functions. This observation was also consistent with the h
    
[^6]: 左/右脑、人类运动控制及对机器人的影响

    Left/Right Brain, human motor control and the implications for robotics. (arXiv:2401.14057v1 [cs.RO])

    [http://arxiv.org/abs/2401.14057](http://arxiv.org/abs/2401.14057)

    本研究通过训练不同的损失函数，实现了类似于人类的左右半球专门化控制系统，该系统在不同的运动任务中展现出协调性、运动效率和位置稳定性的优势。

    

    神经网络运动控制器相对传统控制方法具有各种优点，然而由于其无法产生可靠的精确运动，因此尚未得到广泛采用。本研究探讨了一种双侧神经网络架构作为运动任务的控制系统。我们旨在实现类似于人类在不同任务中观察到的半球专门化：优势系统（通常是右手、左半球）擅长协调和运动效率的任务，而非优势系统在需要位置稳定性的任务上表现更好。通过使用不同的损失函数对半球进行训练，实现了专门化。我们比较了具有专门化半球和无专门化半球、具有半球间连接（代表生物学脑桥）和无半球间连接、具有专门化和无专门化的单侧模型。

    Neural Network movement controllers promise a variety of advantages over conventional control methods however they are not widely adopted due to their inability to produce reliably precise movements. This research explores a bilateral neural network architecture as a control system for motor tasks. We aimed to achieve hemispheric specialisation similar to what is observed in humans across different tasks; the dominant system (usually the right hand, left hemisphere) excels at tasks involving coordination and efficiency of movement, and the non-dominant system performs better at tasks requiring positional stability. Specialisation was achieved by training the hemispheres with different loss functions tailored toward the expected behaviour of the respective hemispheres. We compared bilateral models with and without specialised hemispheres, with and without inter-hemispheric connectivity (representing the biological Corpus Callosum), and unilateral models with and without specialisation. 
    

