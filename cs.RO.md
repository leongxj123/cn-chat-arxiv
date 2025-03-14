# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CMP: Cooperative Motion Prediction with Multi-Agent Communication](https://arxiv.org/abs/2403.17916) | 该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。 |
| [^2] | [Building Cooperative Embodied Agents Modularly with Large Language Models](https://arxiv.org/abs/2307.02485) | 利用大型语言模型构建模块化的协作体现智能体，实现多智能体合作解决具有挑战性的任务，超越规划方法并展示有效沟通。 |

# 详细

[^1]: CMP：具有多智能体通信的合作运动预测

    CMP: Cooperative Motion Prediction with Multi-Agent Communication

    [https://arxiv.org/abs/2403.17916](https://arxiv.org/abs/2403.17916)

    该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。

    

    随着自动驾驶车辆（AVs）的发展和车联网（V2X）通信的成熟，合作连接的自动化车辆（CAVs）的功能变得可能。本文基于合作感知，探讨了合作运动预测的可行性和有效性。我们的方法CMP以LiDAR信号作为输入，以增强跟踪和预测能力。与过去专注于合作感知或运动预测的工作不同，我们的框架是我们所知的第一个解决CAVs在感知和预测模块中共享信息的统一问题。我们的设计中还融入了能够容忍现实V2X带宽限制和传输延迟的独特能力，同时处理庞大的感知表示。我们还提出了预测聚合模块，统一了预测

    arXiv:2403.17916v1 Announce Type: cross  Abstract: The confluence of the advancement of Autonomous Vehicles (AVs) and the maturity of Vehicle-to-Everything (V2X) communication has enabled the capability of cooperative connected and automated vehicles (CAVs). Building on top of cooperative perception, this paper explores the feasibility and effectiveness of cooperative motion prediction. Our method, CMP, takes LiDAR signals as input to enhance tracking and prediction capabilities. Unlike previous work that focuses separately on either cooperative perception or motion prediction, our framework, to the best of our knowledge, is the first to address the unified problem where CAVs share information in both perception and prediction modules. Incorporated into our design is the unique capability to tolerate realistic V2X bandwidth limitations and transmission delays, while dealing with bulky perception representations. We also propose a prediction aggregation module, which unifies the predict
    
[^2]: 用大型语言模型模块化地构建协作体现智能体

    Building Cooperative Embodied Agents Modularly with Large Language Models

    [https://arxiv.org/abs/2307.02485](https://arxiv.org/abs/2307.02485)

    利用大型语言模型构建模块化的协作体现智能体，实现多智能体合作解决具有挑战性的任务，超越规划方法并展示有效沟通。

    

    在这项工作中，我们处理具有去中心化控制、原始感知观察、昂贵通讯和多目标任务的具有各种体现环境的具有挑战性的多智能体合作问题。与先前研究不同的是，我们利用大型语言模型的常识知识、推理能力、语言理解和文本生成能力，并将它们无缝地融入到一个与感知、记忆和执行相结合的认知启发式模块化框架中。从而构建了一个可以规划、沟通和与其他人合作以高效完成长时程任务的合作体现智能体CoELA。我们在C-WAH和TDW-MAT上的实验表明，由GPT-4驱动的CoELA可以超越强大的基于规划的方法，并展示出新兴的有效沟通。

    arXiv:2307.02485v2 Announce Type: replace  Abstract: In this work, we address challenging multi-agent cooperation problems with decentralized control, raw sensory observations, costly communication, and multi-objective tasks instantiated in various embodied environments. While previous research either presupposes a cost-free communication channel or relies on a centralized controller with shared observations, we harness the commonsense knowledge, reasoning ability, language comprehension, and text generation prowess of LLMs and seamlessly incorporate them into a cognitive-inspired modular framework that integrates with perception, memory, and execution. Thus building a Cooperative Embodied Language Agent CoELA, who can plan, communicate, and cooperate with others to accomplish long-horizon tasks efficiently. Our experiments on C-WAH and TDW-MAT demonstrate that CoELA driven by GPT-4 can surpass strong planning-based methods and exhibit emergent effective communication. Though current O
    

