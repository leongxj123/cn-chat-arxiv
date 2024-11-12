# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Importance Guided Data Augmentation for Neural-Based Code Understanding](https://arxiv.org/abs/2402.15769) | 引入了一个通用数据增强框架GenCode，通过重要性指标选择生成的代码作为训练数据，以增强代码理解模型的训练。 |
| [^2] | [MacroSwarm: A Field-based Compositional Framework for Swarm Programming.](http://arxiv.org/abs/2401.10969) | MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。 |

# 详细

[^1]: 重点引导的数据增强用于基于神经网络的代码理解

    Importance Guided Data Augmentation for Neural-Based Code Understanding

    [https://arxiv.org/abs/2402.15769](https://arxiv.org/abs/2402.15769)

    引入了一个通用数据增强框架GenCode，通过重要性指标选择生成的代码作为训练数据，以增强代码理解模型的训练。

    

    arXiv:2402.15769v1 类型：交叉 摘要：预训练的代码模型开启了代码智能时代。最近许多模型都表现出色。然而，在代码学习领域，一个重要问题是自动进行代码数据增强，以帮助开发者准备训练数据，这方面的研究尚不足。本文介绍了一个通用的数据增强框架GenCode，用于增强代码理解模型的训练。GenCode遵循一种生成和选择的范式来准备有用的训练代码。具体来说，它使用代码转换技术首先生成新的代码候选，然后通过重要性指标选择重要的代码作为训练数据。为了评估GenCode与通用重要性指标（损失值）的有效性，我们在四个代码理解任务（如代码克隆检测）和三个预训练代码模型（如CodeT5）上进行实验。与最先进的代码增强技术相比，

    arXiv:2402.15769v1 Announce Type: cross  Abstract: Pre-trained code models lead the era of code intelligence. Many models have been designed with impressive performance recently. However, one important problem, data augmentation for code data that automatically helps developers prepare training data lacks study in the field of code learning. In this paper, we introduce a general data augmentation framework, GenCode, to enhance the training of code understanding models. GenCode follows a generation-and-selection paradigm to prepare useful training codes. Specifically, it uses code transformation techniques to generate new code candidates first and then selects important ones as the training data by importance metrics. To evaluate the effectiveness of GenCode with a general importance metric -- loss value, we conduct experiments on four code understanding tasks (e.g., code clone detection) and three pre-trained code models (e.g., CodeT5). Compared to the state-of-the-art (SOTA) code augm
    
[^2]: MacroSwarm: 一种基于场的组合框架用于群体编程

    MacroSwarm: A Field-based Compositional Framework for Swarm Programming. (arXiv:2401.10969v1 [cs.AI])

    [http://arxiv.org/abs/2401.10969](http://arxiv.org/abs/2401.10969)

    MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。

    

    群体行为工程是一项旨在研究协调简单智能体团体内计算和行动的方法和技术，以实现复杂的全局目标，如图案形成、集体移动、聚类和分布式感知。尽管在群体（无人机、机器人、车辆）分析和工程方面取得了一些进展，但仍然需要通用的设计和实现方法和工具，以系统化的方式定义复杂的群体行为。为了对此做出贡献，本文提出了一种新的基于场的协调方法，称为MacroSwarm，以可重用且完全可组合的功能模块为基础，嵌入集体计算和协调。基于集成计算的宏编程范式，MacroSwarm提出了将每个群体行为块表示为将感知场映射为执行目标场的纯函数的思路。

    Swarm behaviour engineering is an area of research that seeks to investigate methods and techniques for coordinating computation and action within groups of simple agents to achieve complex global goals like pattern formation, collective movement, clustering, and distributed sensing. Despite recent progress in the analysis and engineering of swarms (of drones, robots, vehicles), there is still a need for general design and implementation methods and tools that can be used to define complex swarm behaviour in a principled way. To contribute to this quest, this article proposes a new field-based coordination approach, called MacroSwarm, to design and program swarm behaviour in terms of reusable and fully composable functional blocks embedding collective computation and coordination. Based on the macroprogramming paradigm of aggregate computing, MacroSwarm builds on the idea of expressing each swarm behaviour block as a pure function mapping sensing fields into actuation goal fields, e.g.
    

